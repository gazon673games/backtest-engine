from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import polars as pl
import pytest
from moex_risk_fixture import audit, frame, history, registry, request
from test_intraday_event_support import make_config, write_intraday_dataset

from backtest_engine.accounting.futures import FuturesPortfolioSpec, apply_contract_risk
from backtest_engine.adapters.moex.pipeline.historical_registry import enrich_historical_registry
from backtest_engine.adapters.moex.pipeline.risk_estimates import margin_schedule
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.intraday.event.config import IntradayEventInstrumentSpec
from backtest_engine.intraday.event.data import instrument_spec_for_row, resolve_event_config
from backtest_engine.intraday.event.engine import run_intraday_event_backtest
from backtest_engine.intraday.event.futures import futures_instrument_spec

MODULE = "backtest_engine.adapters.moex.pipeline.historical_registry"


def enrich(tmp_path, monkeypatch, **options):
    monkeypatch.setattr(MODULE + ".download_risk_history", lambda *args: history())
    return enrich_historical_registry(
        request(tmp_path, **options), tmp_path / "reports", frame(), registry(), audit()
    )


def test_estimates_replace_current_margin_and_are_audited(tmp_path, monkeypatch):
    result, report = enrich(tmp_path, monkeypatch)
    item = result["instruments"][0]
    assert item["initial_margin"] is None, "current GO must not become historical GO"
    assert item["rule_source"] == "ncc_mr1_notional_estimate_v1"
    assert item["margin_estimate_schedule"][0]["rate"] == 0.08
    assert report["status"] == "warning"
    assert "Research margin estimate" in " ".join(report["warnings"])
    assert report["historical_margin"]["missing_coverage"] == []


def test_strict_build_does_not_invent_exact_margin(tmp_path, monkeypatch):
    result, report = enrich(tmp_path, monkeypatch, rules_enforcement="strict")
    assert not result["instruments"][0]["margin_estimate_schedule"]
    assert report["status"] == "warning"
    assert report["historical_margin"]["missing_coverage"]


def test_exact_dated_rules_override_estimates_without_network(tmp_path, monkeypatch):
    path = tmp_path / "rules.json"
    path.write_text(
        json.dumps(
            {
                "instruments": [
                    {
                        "symbol": "CRH6",
                        "initial_margin": 1200,
                        "valid_from": "2026-01-01",
                        "valid_to": "2026-03-19",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        MODULE + ".download_risk_history",
        lambda *_: pytest.fail("exact rules do not require risk downloads"),
    )
    result, report = enrich_historical_registry(
        request(tmp_path, rules_path=path, cache_policy="offline", rules_enforcement="strict"),
        tmp_path / "reports",
        frame(),
        registry(),
        audit(),
    )
    assert report["status"] == "approved"
    assert result["instruments"][0]["initial_margin"] == 1200
    assert result["instruments"][0]["contract_multiplier"] == 1000
    assert report["historical_margin"]["exact_rules"]["sha256"]


def test_network_failure_remains_visible_and_blocks_missing_coverage(tmp_path, monkeypatch):
    def unavailable(*_):
        raise OSError("ISS unavailable")

    monkeypatch.setattr(MODULE + ".download_risk_history", unavailable)
    result, report = enrich_historical_registry(
        request(tmp_path), tmp_path / "reports", frame(), registry(), audit()
    )
    assert report["status"] == "warning"
    assert report["historical_margin"]["missing_coverage"]
    assert "ISS unavailable" in " ".join(report["warnings"])
    assert result["instruments"][0]["initial_margin"] is None


def native_config(tmp_path):
    path = write_intraday_dataset(
        tmp_path,
        timestamps=[datetime(2026, 1, 12, 15) + timedelta(minutes=5 * i) for i in range(5)],
        closes=[100, 100, 100, 102, 102],
        active_contracts=["CRH6"] * 5,
    )
    plugin = tmp_path / "strategy.py"
    plugin.write_text(
        "def initialize(params):\n    return {'calls': 0}\n"
        "def generate_signal(ctx, params):\n"
        "    ctx.state['calls'] += 1\n"
        "    return {'target': 1, 'contracts': 1 if ctx.state['calls'] == 1 else 2,\n"
        "            'skip_unaffordable_add': True, 'reason': 'test_add'}\n"
    )
    schedule = margin_schedule(history(), [date(2026, 1, 12)], 1000)
    instrument = IntradayEventInstrumentSpec(
        symbol="CRH6",
        contract_multiplier=1000,
        price_currency="RUB",
        pnl_currency="RUB",
        margin_estimate_schedule=schedule,
    )
    return make_config(
        tmp_path,
        path,
        plugin,
        instrument_specs=[instrument],
        initial_capital=12000,
        sizing_mode="strategy_contracts",
        execution_timing="signal_close_to_next_bar_close",
        max_margin_usage=1,
    )


def test_native_addition_uses_historical_estimate_and_reconciles_ledger(tmp_path):
    config = native_config(tmp_path)
    output = run_intraday_event_backtest(config).output_dir
    events = pl.read_parquet(output / "events.parquet")
    ledger = pl.read_parquet(output / "portfolio_ledger.parquet")
    assert "insufficient_margin_add_skipped" in events["strategy_reason"].to_list()
    assert events["target_contracts"].max() == 1
    assert ledger["equity_cash"][-1] == pytest.approx(14000)
    assert ledger["reconciliation_diff"].abs().max() == pytest.approx(0, abs=1e-6)
    saved = json.loads((output / "config.json").read_text("utf-8"))
    assert saved["instrument"]["specs"][0]["margin_estimate_schedule"]


def test_strict_runtime_rejects_estimate_even_if_approval_was_claimed(tmp_path):
    config = native_config(tmp_path)
    config.dataset.quality_approval_required = True
    with pytest.raises(DatasetValidationError, match="Research margin estimates"):
        resolve_event_config(config, {"bar_minutes": 5})


def test_first_concentration_tier_is_an_actual_risk_limit(tmp_path):
    config = native_config(tmp_path)
    spec = instrument_spec_for_row(
        {"active_contract": "CRH6", "close": 100}, datetime(2026, 1, 12, 15), config.instrument
    )
    result = apply_contract_risk(
        contracts=2_000_000,
        fill_price=100,
        instrument=futures_instrument_spec(spec),
        portfolio=FuturesPortfolioSpec(),
        equity_cash=1e12,
    )
    assert result.status == "rejected"
    assert result.max_contracts == 1_500_000


def test_strict_catalog_also_rejects_estimates(tmp_path):
    config = native_config(tmp_path)
    config.dataset.catalog_enforcement = "strict"
    with pytest.raises(DatasetValidationError, match="Research margin estimates"):
        resolve_event_config(config, {"bar_minutes": 5})


def test_incomplete_registry_blocks_before_any_strategy_is_called(tmp_path, monkeypatch):
    from backtest_engine.intraday.event.config import IntradayEventInstrumentConfig
    from backtest_engine.intraday.event.data import resolve_instrument_config

    result, _ = enrich(tmp_path, monkeypatch, rules_enforcement="strict")
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(result))
    with pytest.raises(DatasetValidationError, match="Historical margin coverage is incomplete"):
        resolve_instrument_config(IntradayEventInstrumentConfig(registry_path=path))


def test_conflicting_exact_rule_intervals_are_rejected(tmp_path):
    from backtest_engine.adapters.moex.pipeline.historical_registry import load_exact_rules

    record = {
        "symbol": "CRH6",
        "initial_margin": 1000,
        "valid_from": "2026-01-01",
        "valid_to": "2026-03-19",
    }
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"instruments": [record, {**record, "initial_margin": 2000}]}))
    with pytest.raises(ValueError, match="Overlapping exact MOEX rules for CRH6"):
        load_exact_rules(path)
