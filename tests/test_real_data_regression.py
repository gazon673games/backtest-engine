from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.intraday.event.engine import (
    IntradayEventBacktestConfig,
    IntradayEventDatasetConfig,
    IntradayEventExecutionConfig,
    IntradayEventOutputConfig,
    IntradayEventPortfolioConfig,
    IntradayEventStrategyConfig,
    run_intraday_event_backtest,
)
from backtest_engine.research.external_validation import (
    ExternalValidationConfig,
    validate_against_statement,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "real" / "cnyrub_nearest_5m_2024-07-01_1330.csv"
STATEMENT_PATH = (
    Path(__file__).parent / "fixtures" / "real" / "cnyrub_hand_calculated_statement_2024-07-01.csv"
)


def test_golden_real_cnyrub_nearest_5m_strict_catalog_regression(tmp_path: Path) -> None:
    frame = pl.read_csv(FIXTURE_PATH, try_parse_dates=True).with_columns(
        pl.col("currency").fill_null("RUB")
    )
    dataset_path = tmp_path / "cnyrub_nearest_5m.parquet"
    frame.write_parquet(dataset_path)
    plugin_path = tmp_path / "always_long.py"
    plugin_path.write_text("def generate_signal(ctx):\n    return 1.0\n", encoding="utf-8")
    catalog_path = write_catalog_bundle(tmp_path, dataset_path)

    result = run_intraday_event_backtest(
        IntradayEventBacktestConfig(
            dataset=IntradayEventDatasetConfig(
                path=dataset_path,
                catalog_path=catalog_path,
                dataset_id="golden:cnyrub-nearest-5m:2024-07-01:1330",
                catalog_enforcement="strict",
                quality_approval_required=True,
                allow_legacy_bar_timestamps=True,
            ),
            strategy=IntradayEventStrategyConfig(strategy_file=plugin_path),
            portfolio=IntradayEventPortfolioConfig(
                initial_capital=1000.0,
                sizing_mode="fixed_contracts",
                fixed_contracts=1.0,
            ),
            execution=IntradayEventExecutionConfig(
                timing="signal_close_to_same_close",
                allow_noncausal_same_close=True,
                end_of_data_position_policy="mark",
                from_date=date(2024, 7, 1),
            ),
            output=IntradayEventOutputConfig(root=tmp_path / "out", run_id="golden_real"),
        )
    )

    events = pl.read_parquet(result.output_dir / "events.parquet")
    trades = pl.read_parquet(result.output_dir / "trades.parquet")
    ledger = pl.read_parquet(result.output_dir / "portfolio_ledger.parquet")
    manifest = json.loads(
        (result.output_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )

    assert result.catalog_approval["status"] == "approved"
    assert result.quality_approval["status"] == "approved"
    assert result.dataset_validation["continuous_symbol"] == "CNYRUB_NEAREST_5M"
    assert events.height == 7
    # CR has a 0.001 price tick worth 1 RUB, so one full price point is 1000 RUB.
    assert events.get_column("gross_pnl").to_list() == pytest.approx(
        [1.0, 0.0, -11.0, 11.0, -5.0, 4.0, -12.0]
    )
    assert result.metrics["order_count"] == 1
    assert result.metrics["filled_order_count"] == 1
    assert result.metrics["net_pnl"] == pytest.approx(-12.0)
    assert result.metrics["final_equity_cash"] == pytest.approx(988.0)
    assert events.get_column("net_pnl").sum() == pytest.approx(
        result.metrics["final_equity_cash"] - 1000.0
    )
    assert trades.get_column("net_pnl").sum() == pytest.approx(events.get_column("net_pnl").sum())
    assert ledger.get_column("equity_cash")[-1] == pytest.approx(
        result.metrics["final_equity_cash"]
    )
    assert ledger.get_column("reconciliation_diff").abs().max() == pytest.approx(0.0)
    assert manifest["dataset"]["sha256"] == file_sha256(dataset_path)
    assert manifest["strategy"]["sha256"] == file_sha256(plugin_path)
    assert manifest["outputs"]["events.parquet"]["sha256"] == file_sha256(
        result.output_dir / "events.parquet"
    )
    external = validate_against_statement(
        ExternalValidationConfig(
            backtest_events_path=result.output_dir / "events.parquet",
            statement_path=STATEMENT_PATH,
            output_path=tmp_path / "external_validation.json",
            tolerance=1e-9,
        )
    )
    assert external.status == "approved"
    assert external.break_count == 0


def write_catalog_bundle(tmp_path: Path, dataset_path: Path) -> Path:
    registry_path = tmp_path / "instrument_registry.json"
    calendar_path = tmp_path / "session_calendar.json"
    approval_path = tmp_path / "quality_approval.json"
    catalog_path = tmp_path / "data_catalog.json"
    dataset_sha = file_sha256(dataset_path)
    registry_path.write_text(
        json.dumps(
            {
                "instruments": [
                    {
                        "symbol": "CR*",
                        "price_currency": "RUB",
                        "pnl_currency": "RUB",
                        "contract_multiplier": 1.0,
                        "lot_size": 1.0,
                        "tick_size": 0.001,
                        "tick_value": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calendar_path.write_text(
        json.dumps(
            {
                "main_session_start": "10:00",
                "main_session_end": "18:50",
                "evening_session_start": "19:00",
                "evening_session_end": "23:50",
                "holiday_dates": [],
                "short_day_dates": [],
            }
        ),
        encoding="utf-8",
    )
    approval_path.write_text(
        json.dumps(
            {
                "approval_status": "approved",
                "continuous_symbol": "CNYRUB_NEAREST_5M",
                "timeframe": "5m",
                "dataset_sha256": dataset_sha,
                "usable_decision_day_pct": 1.0,
            }
        ),
        encoding="utf-8",
    )
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_version": 1,
                "entries": [
                    {
                        "dataset_id": "golden:cnyrub-nearest-5m:2024-07-01:1330",
                        "dataset_path": str(dataset_path),
                        "dataset_sha256": dataset_sha,
                        "quality_approval_path": str(approval_path),
                        "instrument_registry_path": str(registry_path),
                        "calendar_path": str(calendar_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return catalog_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
