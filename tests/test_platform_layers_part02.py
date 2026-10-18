from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.accounting.ledger import (
    CommissionSchedule,
    MultiCurrencyLedger,
)
from backtest_engine.adapters.moex.iss.snapshot import (
    MoexIssSnapshotConfig,
    build_moex_iss_snapshot,
)
from backtest_engine.adapters.moex.registry import (
    MoexRegistryUpdateConfig,
    update_moex_registry_and_calendar,
)
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.store import (
    SchemaMigrationRecord,
    add_schema_migration,
    build_data_store_from_catalog,
    canonical_dataset_path,
    load_data_store_manifest,
    parquet_metadata_index,
)
from backtest_engine.research.validation.config import ResearchValidationConfig
from backtest_engine.research.validation.gate import evaluate_research_gate
from backtest_engine.strategies.plugins.loader import load_intraday_strategy_plugin
from backtest_engine.strategies.runtime import (
    StrategyMetadata,
    StrategyParamSpec,
    StrategyRuntimeContext,
    strategy_fingerprint,
    validate_strategy_params,
)


def test_data_store_indexes_catalog_versions_and_schema_migrations(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "dataset_id": "moex:cnyrub:5m",
                        "dataset_path": str(dataset_path),
                        "metadata": {"roll_rule": "nearest"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    store_path = tmp_path / "store.json"

    manifest = build_data_store_from_catalog(
        catalog_path=catalog_path,
        output_path=store_path,
        adapter_snapshot={"adapter_name": "fixture", "adapter_version": "1"},
    )
    add_schema_migration(
        store_path,
        SchemaMigrationRecord(
            dataset_id="moex:cnyrub:5m",
            from_schema_version=1,
            to_schema_version=2,
            migration_name="add_currency",
        ),
    )
    reloaded = load_data_store_manifest(store_path)

    assert len(manifest.versions) == 1
    assert reloaded.versions[0].dataset_id == "moex:cnyrub:5m"
    assert reloaded.versions[0].lineage.adapter_snapshot["adapter_name"] == "fixture"
    assert reloaded.schema_migrations[0].migration_name == "add_currency"


def test_data_store_canonical_path_and_parquet_metadata_index(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    canonical = canonical_dataset_path(
        root=tmp_path / "store",
        dataset_id="moex:cny/rub",
        stage="bars",
        version_id="v1",
    )
    index = parquet_metadata_index(dataset_path)

    assert canonical == tmp_path / "store" / "moex_cny_rub" / "bars" / "v1" / "dataset.parquet"
    assert index["num_rows"] == 4
    assert "close" in index["schema"]


def test_moex_registry_update_builds_observed_specs_calendar_and_audit(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    result = update_moex_registry_and_calendar(
        MoexRegistryUpdateConfig(
            dataset_paths=[dataset_path],
            instrument_output_path=tmp_path / "instruments.json",
            calendar_output_path=tmp_path / "calendar.json",
            audit_output_path=tmp_path / "audit.json",
            tick_size=0.001,
            tick_value=1.0,
        )
    )
    instruments = json.loads(result.instrument_output_path.read_text(encoding="utf-8"))
    calendar = json.loads(result.calendar_output_path.read_text(encoding="utf-8"))
    audit = json.loads(result.audit_output_path.read_text(encoding="utf-8"))

    assert result.instrument_count == 1
    assert instruments["instruments"][0]["symbol"] == "CRU4"
    assert "2024-01-03" in calendar["unobserved_weekdays"]
    assert "2024-01-04" in calendar["short_session_candidates"]
    assert audit["status"] == "warning"


def test_moex_iss_snapshot_writes_authoritative_artifacts(tmp_path: Path) -> None:
    raw_path = tmp_path / "iss_raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "securities": {
                    "columns": ["SECID", "LOTSIZE", "MINSTEP", "CURRENCYID"],
                    "data": [["CRU4", 1, 0.001, "RUB"]],
                },
                "calendar": {"trading_dates": ["2024-01-02"]},
            }
        ),
        encoding="utf-8",
    )

    result = build_moex_iss_snapshot(
        MoexIssSnapshotConfig(
            raw_snapshot_path=raw_path,
            snapshot_output_path=tmp_path / "snapshot.json",
            audit_output_path=tmp_path / "audit.json",
            registry_output_path=tmp_path / "registry.json",
            calendar_output_path=tmp_path / "calendar.json",
        )
    )
    registry = json.loads(result.registry_output_path.read_text(encoding="utf-8"))

    assert result.status == "approved"
    assert registry["source"] == "moex_iss"
    assert registry["instruments"][0]["symbol"] == "CRU4"


def test_multi_currency_ledger_tracks_cash_commissions_fx_and_reconciliation() -> None:
    ledger = MultiCurrencyLedger(base_currency="RUB")
    ts = datetime(2024, 1, 2, 10, 0)

    ledger.deposit(ts=ts, currency="RUB", amount=1000.0)
    ledger.deposit(ts=ts, currency="USD", amount=10.0)
    ledger.apply_trade(
        ts=ts,
        symbol="CRU4",
        side="buy",
        quantity=2.0,
        price=100.0,
        currency="RUB",
        commission=CommissionSchedule(currency="RUB", fixed_per_order=1.0),
    )
    ledger.apply_variation_margin(ts=ts, symbol="CRU4", amount=5.0, currency="RUB")

    assert ledger.cash["RUB"] == pytest.approx(804.0)
    assert ledger.positions["CRU4"].quantity == pytest.approx(2.0)
    assert ledger.equity({("USD", "RUB"): 90.0}) == pytest.approx(1704.0)
    assert ledger.reconcile_cash({"RUB": 804.0, "USD": 9.0})[0].key == "cash:USD"


def test_strategy_runtime_validates_params_warmup_and_lookahead(tmp_path: Path) -> None:
    frame = pl.DataFrame({"close": [10.0, 11.0, 12.0]})
    ctx = StrategyRuntimeContext(frame, current_index=1, warmup_bars=2)
    specs = (
        StrategyParamSpec("window", "int", min_value=1, max_value=100, required=True),
        StrategyParamSpec("enabled", "bool", default=True),
    )
    strategy_path = tmp_path / "strategy.py"
    strategy_path.write_text("def generate_signal(ctx):\n    return 1.0\n", encoding="utf-8")

    params = validate_strategy_params({"window": "5"}, specs)
    fingerprint = strategy_fingerprint(
        strategy_path=strategy_path,
        metadata=StrategyMetadata(name="fixture", params=specs),
        params=params,
    )

    ctx.require_warmup()
    assert params == {"window": 5, "enabled": True}
    assert ctx.value("close") == pytest.approx(11.0)
    assert len(fingerprint) == 64
    with pytest.raises(DatasetValidationError, match="future data"):
        ctx.value("close", offset=1)


def test_plugin_loader_enforces_declared_strategy_metadata(tmp_path: Path) -> None:
    plugin_path = tmp_path / "strict_plugin.py"
    plugin_path.write_text(
        """
STRATEGY_METADATA = {
    "name": "strict",
    "warmup_bars": 2,
    "params": [{"name": "window", "param_type": "int", "required": True, "min_value": 1}],
}

def generate_signal(ctx, params=None):
    return {"target": 1.0, "features": {"window": params["window"]}}
""".lstrip(),
        encoding="utf-8",
    )
    generate_signal = load_intraday_strategy_plugin(plugin_path)
    frame = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 2, 10, 0), datetime(2024, 1, 2, 10, 5)],
            "timeframe": ["5m", "5m"],
            "close": [1.0, 2.0],
        }
    )
    from backtest_engine.strategies.intraday_api import IntradayStrategyContext
    from backtest_engine.strategies.plugins.loader import (
        generate_intraday_event_plugin_decision_from_callable,
    )

    ctx = IntradayStrategyContext(frame, end_index=1)
    decision = generate_intraday_event_plugin_decision_from_callable(
        generate_signal,
        ctx,
        {"window": "3"},
    )

    assert decision.features["window"] == 3
    with pytest.raises(Exception, match="Unknown strategy params"):
        generate_intraday_event_plugin_decision_from_callable(
            generate_signal,
            ctx,
            {"window": "3", "oops": "1"},
        )


def test_research_gate_rejects_guardrail_failures(tmp_path: Path) -> None:
    guardrails = pl.DataFrame(
        [
            {
                "multiple_testing_warning": False,
                "rank_instability_warning": True,
                "oos_degradation_warning": False,
                "negative_oos_run_share": 0.75,
            }
        ]
    )

    gate = evaluate_research_gate(
        ResearchValidationConfig(
            experiment_path=tmp_path,
            max_negative_oos_run_share=0.5,
        ),
        guardrails,
        warnings=[],
    )

    assert gate["acceptance_status"] == "rejected"
    assert "rank_instability_warning" in gate["reasons"]


def write_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.parquet"
    pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 2, 10, 0),
                datetime(2024, 1, 2, 18, 50),
                datetime(2024, 1, 4, 10, 0),
                datetime(2024, 1, 4, 12, 0),
            ],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [10.0, 20.0, 30.0, 40.0],
            "source_contract_symbol": ["CRU4", "CRU4", "CRU4", "CRU4"],
        }
    ).write_parquet(path)
    return path
