"""Offline portfolio API, frozen inputs, native ledger and end-date contracts."""

import json
from datetime import timedelta

import polars as pl
import pytest
from test_sdk_portfolio_native import bars

from backtest_engine.workbench.history import history_path_for, load_workbench_history
from backtest_engine.workbench.portfolio_api import (
    PortfolioRequest,
    freeze_portfolio,
    portfolio_list_payload,
    portfolio_save_payload,
)
from backtest_engine.workbench.portfolio_jobs import execute_portfolio
from backtest_engine.workbench.report import build_workbench_report
from backtest_engine.workbench.server import get_payload, post_payload


@pytest.fixture
def portfolio_payload(tmp_path):
    source = tmp_path / "fixture_strategy.py"
    source.write_text(
        """from strategy_sdk import (
    DataRequirements, StrategyMetadata, long_decision, hold_decision,
)
STRATEGY_METADATA = StrategyMetadata("fixture")
def data_requirements(params):
    return DataRequirements("M60", 1)
def generate_signal(ctx, params):
    if ctx.position_quantity:
        return hold_decision(reason="keep")
    return long_decision(contracts=3, reason="entry")
""",
        encoding="utf-8",
    )
    legs, sleeves = [], []
    for symbol, price in (("A", 100), ("B", 50)):
        path = tmp_path / f"{symbol}.parquet"
        frame = bars(symbol, price)
        # Future extreme prices must not influence a run ending on Jan 1.
        future = frame.with_columns(
            [
                pl.col(name) + timedelta(days=1)
                for name in ("ts", "bar_open_ts", "bar_close_ts", "available_at")
            ]
        ).with_columns(
            [
                pl.col(name) * 10
                for name in ("open", "high", "low", "close", "mark_price", "mark_low", "mark_high")
            ]
        )
        pl.concat([frame, future]).write_parquet(path)
        legs.append(
            {
                "asset_id": symbol,
                "dataset_path": str(path),
                "quality_approval_required": False,
                "asset_class": "crypto",
                "instrument_type": "perpetual",
                "settlement_currency": "USDT",
                "price_currency": "USDT",
                "pnl_currency": "USDT",
                "quantity_step": 0.001,
                "minimum_quantity": 0.001,
                "taker_fee_bps": 5,
            }
        )
        sleeves.append(
            {
                "asset_id": symbol,
                "strategy_file": str(source),
                "initial_capital": 300,
                "max_exposure": 2,
            }
        )
    return {
        "name": "Golden portfolio",
        "diagnostic": True,
        "config": {
            "legs": legs,
            "strategy": {"sdk_sleeves": sleeves, "strict_runtime": True},
            "portfolio": {"initial_capital": 700, "base_currency": "USDT", "leverage": 3},
            "execution": {
                "from_date": "2025-01-01",
                "till_date": "2025-01-01",
                "timing": "signal_close_to_next_bar_open_to_close",
                "end_of_data_position_policy": "mark",
                "trading_days_per_year": 365,
            },
            "output": {"root": str(tmp_path / "ignored"), "run_id": "portfolio-test"},
        },
    }


def test_native_frozen_portfolio_report_and_history(tmp_path, portfolio_payload):
    root = tmp_path / "runs"
    frozen = freeze_portfolio(root, portfolio_payload)
    source = portfolio_payload["config"]["strategy"]["sdk_sleeves"][0]["strategy_file"]
    from pathlib import Path

    Path(source).write_text("raise RuntimeError('changed source')", encoding="utf-8")
    execute_portfolio("portfolio-test", frozen)
    report = get_payload(root, "/api/runs/portfolio-test/report", "")
    # A=3 units -> +30, B=3 units -> +15; entry fees=.225, funding=4.95.
    assert report["metrics"]["final_equity_cash"] == pytest.approx(739.825)
    assert report["kind"] == "portfolio"
    assert {asset["asset_id"] for asset in report["assets"]} == {"A", "B"}
    assert all(asset["fill_count"] == 1 for asset in report["assets"])
    assert all(len(asset["fills"]) == 1 for asset in report["assets"])
    assert max(row["ts"] for row in report["series"]["equity"]).startswith("2025-01-01")
    ledger = pl.read_parquet(root / "portfolio-test/portfolio_ledger.parquet")
    assert ledger["equity_cash"][-1] == report["series"]["equity"][-1]["equity_cash"]
    assert ledger["reconciliation_diff"].abs().max() < 1e-8
    assert build_workbench_report(root / "portfolio-test", max_points=2)["kind"] == "portfolio"
    history = load_workbench_history(history_path_for(root))["runs"][0]
    assert history["kind"] == "portfolio" and history["status"] == "completed"
    assert history["fingerprint"] == frozen["fingerprint"]
    changed = Path(portfolio_payload["config"]["legs"][0]["dataset_path"])
    changed.write_bytes(changed.read_bytes() + b"changed")
    historical = build_workbench_report(root / "portfolio-test")
    assert historical["metrics"] == report["metrics"]
    assert historical["assets"][0]["chart_warning"]
    assert historical["assets"][0]["candles"] == []


def test_changed_input_rejected_and_failure_preserved(tmp_path, portfolio_payload):
    root = tmp_path / "runs"
    frozen = freeze_portfolio(root, portfolio_payload)
    from pathlib import Path

    path = Path(portfolio_payload["config"]["legs"][0]["dataset_path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="input changed"):
        execute_portfolio("portfolio-test", frozen)
    assert load_workbench_history(history_path_for(root))["runs"][0]["status"] == "failed"


def test_explicit_quality_gates_and_invalid_budget(portfolio_payload):
    portfolio_payload["diagnostic"] = False
    with pytest.raises(ValueError, match="diagnostic=true"):
        PortfolioRequest.model_validate(portfolio_payload)
    portfolio_payload["diagnostic"] = True
    portfolio_payload["config"]["portfolio"]["initial_capital"] = 500
    with pytest.raises(ValueError, match="budgets exceed"):
        PortfolioRequest.model_validate(portfolio_payload)


def test_presets_and_shared_queue(tmp_path, portfolio_payload, monkeypatch):
    root = tmp_path / "runs"
    saved = post_payload(root, "/api/portfolios", portfolio_payload)
    assert portfolio_list_payload(root)["items"] == [saved]
    saved["name"] = "Updated"
    portfolio_save_payload(root, saved)
    assert len(get_payload(root, "/api/portfolios", "")["items"]) == 1
    calls = []
    monkeypatch.setattr(
        "backtest_engine.workbench.portfolio_api.submit_job", lambda *args: calls.append(args)
    )
    queued = post_payload(root, "/api/portfolio-runs/async", saved)
    assert calls[0][2] == "portfolio"
    assert queued["fingerprint"] == calls[0][3]["fingerprint"]
    assert json.loads(json.dumps(calls[0][3]))["request"]["config"]["output"]["root"] == str(root)


def test_empty_run_id_is_generated(tmp_path, portfolio_payload):
    portfolio_payload["config"]["output"]["run_id"] = ""
    first = freeze_portfolio(tmp_path, portfolio_payload)
    second = freeze_portfolio(tmp_path, portfolio_payload)
    assert first["request"]["config"]["output"]["run_id"].startswith("portfolio-")
    assert (
        first["request"]["config"]["output"]["run_id"]
        != second["request"]["config"]["output"]["run_id"]
    )


def test_frozen_calendar_blocks_one_sleeve_without_discarding_market_history(
    tmp_path, portfolio_payload
):
    sleeves = portfolio_payload["config"]["strategy"]["sdk_sleeves"]
    sleeves[0]["risk_increase_blocked_weekdays"] = [2]  # Wednesday, Jan 1.
    root = tmp_path / "runs"
    frozen = freeze_portfolio(root, portfolio_payload)
    sleeves[0]["risk_increase_blocked_weekdays"] = []  # Queued policy stays pinned.
    execute_portfolio("portfolio-test", frozen)
    ledger = pl.read_parquet(root / "portfolio-test/portfolio_ledger.parquet")
    legs = pl.read_parquet(root / "portfolio-test/portfolio_legs.parquet")
    assert ledger["equity_cash"][-1] == pytest.approx(713.275)
    assert ledger["reconciliation_diff"].abs().max() < 1e-8
    assert legs.filter(
        (pl.col("asset_id") == "A") & (pl.col("filled_contract_delta") != 0)
    ).is_empty()
    assert (
        legs.filter((pl.col("asset_id") == "B") & (pl.col("filled_contract_delta") != 0)).height
        == 1
    )


@pytest.mark.parametrize("list_registry", [False, True])
def test_portfolio_refreeze_is_idempotent_with_instrument_registry(
    tmp_path, portfolio_payload, list_registry
):
    spec = {"symbol": "A", "quantity_step": 0.001, "taker_fee_bps": 5}
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"instruments": [spec]} if list_registry else spec))
    portfolio_payload["config"]["legs"][0]["instrument_registry_path"] = str(registry)
    root = tmp_path / "runs"
    first = freeze_portfolio(root, portfolio_payload)
    second = freeze_portfolio(root, first["request"])
    assert second["request"] == first["request"]
    assert second["fingerprint"] == first["fingerprint"]


def test_frozen_shifted_signal_executes_after_its_completed_bucket(tmp_path, portfolio_payload):
    from pathlib import Path

    for sleeve in portfolio_payload["config"]["strategy"]["sdk_sleeves"]:
        source = Path(sleeve["strategy_file"])
        source.write_text(
            source.read_text().replace('DataRequirements("M60", 1)', 'DataRequirements("M120", 1)')
        )
        sleeve["signal_offset_minutes"] = 60
    root = tmp_path / "runs"
    frozen = freeze_portfolio(root, portfolio_payload)
    execute_portfolio("portfolio-test", frozen)
    legs = pl.read_parquet(root / "portfolio-test/portfolio_legs.parquet")
    fills = legs.filter(pl.col("filled_contract_delta") != 0)
    assert fills.height == 2
    assert fills["fill_ts"].dt.hour().to_list() == [3, 3]
