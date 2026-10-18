"""Financial checks for persistent, delayed and atomic portfolio orders."""

import json
from datetime import date, datetime, timedelta

import polars as pl
import pytest
from neutral_adapter_fixture import NeutralAdapter

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.portfolio.multi.config import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentExecutionConfig,
    MultiInstrumentLegConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
    MultiInstrumentUniverseConfig,
)
from backtest_engine.portfolio.multi.engine import run_multi_instrument_backtest


def run_orders(
    tmp_path, *, latency=1, close=True, partial=False, atomic=False, source=None, universe=None
):
    dataset = NeutralAdapter().build_dataset(
        DatasetBuildRequest(
            adapter_id=NeutralAdapter.id,
            engine="cash",
            security="GOLDUSD",
            from_date="2024-01-02",
            till_date="2024-01-02",
            output_root=tmp_path / "data",
        )
    )
    spec = json.loads(dataset.instrument_registry_path.read_text())["instruments"][0]
    spec.pop("symbol")
    legs = []
    for symbol in ["A", "B"] if atomic else ["A"]:
        path = tmp_path / f"{symbol}.parquet"
        volume = [0, 0, 0, 1000, 1000, 1000] if symbol == "B" else [1 if partial else 1000] * 6
        pl.read_parquet(dataset.dataset_path).with_columns(
            pl.lit(symbol).alias("symbol"),
            pl.lit(symbol).alias("active_contract"),
            pl.Series("volume", volume),
            pl.lit(symbol).alias("base_currency"),
        ).write_parquet(path)
        legs.append(
            MultiInstrumentLegConfig(
                asset_id=symbol,
                dataset_path=path,
                quality_approval_required=False,
                **{**spec, "base_currency": symbol},
            )
        )
    strategy = tmp_path / "orders.py"
    strategy.write_text(
        "calls = 0\ndef generate_targets(ctx):\n    global calls\n    calls += 1\n"
        "    if calls == 1:\n        return {key: 1 for key in ctx.tradable_assets}\n"
        + (
            "    if calls == 3:\n        return {key: 0 for key in ctx.tradable_assets}\n"
            if close
            else ""
        )
        + "    return {}\n"
    )
    if source is not None:
        strategy.write_text(source)
    return run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=legs,
            strategy=MultiInstrumentStrategyConfig(strategy_file=strategy),
            universe=MultiInstrumentUniverseConfig(path=universe),
            portfolio=MultiInstrumentPortfolioConfig(
                initial_capital=1000, base_currency="USD", fixed_contracts=2.5
            ),
            execution=MultiInstrumentExecutionConfig(
                timing="signal_close_to_next_bar_open_to_close",
                from_date=date(2024, 1, 2),
                latency_bars=latency,
                volume_participation_limit=0.5 if partial else (1.0 if atomic else None),
                rebalance_policy="all_or_none" if atomic else "independent",
                end_of_data_position_policy="mark",
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "runs", run_id="delayed"),
        )
    )


def test_delayed_orders_survive_hold_signals_and_charge_confirmed_fills(tmp_path):
    result = run_orders(tmp_path)
    rows = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    fills = rows.filter(pl.col("filled_contract_delta") != 0)
    assert fills["fill_price"].to_list() == [110, 120]
    assert fills["filled_contract_delta"].to_list() == [2.5, -2.5]
    assert result.metrics["final_equity_cash"] == pytest.approx(1024.425)
    orders = json.loads((result.output_dir / "execution_orders.json").read_text())
    assert [order["status"] for order in orders] == ["filled", "filled"]
    assert result.metrics["pending_orders_cancelled"] == 0


def test_changed_target_cancels_pending_buy_without_fake_fill(tmp_path):
    result = run_orders(tmp_path, latency=2)
    assert result.metrics["final_equity_cash"] == 1000
    orders = json.loads((result.output_dir / "execution_orders.json").read_text())
    assert len(orders) == 1
    assert orders[0]["filled_quantity"] == 0
    assert orders[0]["cancellation_reason"] == "target_replaced"


def test_partial_order_remainder_continues_and_is_cancelled_at_end(tmp_path):
    result = run_orders(tmp_path, close=False, partial=True)
    rows = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    assert rows["filled_contract_delta"].to_list() == [0, 0.5, 0.5, 0.5, 0.5]
    assert result.metrics["final_equity_cash"] == pytest.approx(1004.765)
    assert result.metrics["pending_orders_cancelled"] == 1
    order = json.loads((result.output_dir / "execution_orders.json").read_text())[0]
    assert order["filled_quantity"] == 2
    assert order["cancellation_reason"] == "end_of_data"


def test_atomic_batch_waits_for_both_legs_without_consuming_rejected_fills(tmp_path):
    result = run_orders(tmp_path, close=False, atomic=True)
    rows = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    fills = rows.filter(pl.col("filled_contract_delta") != 0)
    assert fills["asset_id"].to_list() == ["A", "B"]
    assert fills["fill_price"].to_list() == [120, 120]
    assert fills["filled_contract_delta"].to_list() == [2.5, 2.5]
    assert result.metrics["final_equity_cash"] == pytest.approx(999.4)


def test_atomic_rollback_does_not_resurrect_previously_filled_order(tmp_path):
    source = (
        "calls=0\ndef generate_targets(ctx):\n    global calls\n    calls+=1\n"
        '    if calls==1: return {"A": 1}\n'
        '    if calls==3: return {"B": 1}\n    return {}\n'
    )
    result = run_orders(tmp_path, atomic=True, source=source)
    rows = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    fills = rows.filter(pl.col("filled_contract_delta") != 0)
    assert fills["asset_id"].to_list() == ["A", "B"]
    assert fills["filled_contract_delta"].to_list() == [2.5, 2.5]
    assert result.metrics["final_equity_cash"] == pytest.approx(1024.425)


def test_universe_removal_cancels_unfilled_order(tmp_path):
    start = datetime(2024, 1, 2, 1, 5)
    path = tmp_path / "universe.parquet"
    pl.DataFrame(
        {
            "effective_at": [start, start + timedelta(minutes=5)],
            "available_at": [start, start + timedelta(minutes=5)],
            "asset_id": ["A", "A"],
            "status": ["active", "inactive"],
            "quote_volume": [100, 100],
        }
    ).write_parquet(path)
    result = run_orders(tmp_path, latency=2, universe=path)
    assert result.metrics["final_equity_cash"] == 1000
    order = json.loads((result.output_dir / "execution_orders.json").read_text())[0]
    assert order["filled_quantity"] == 0
    assert order["cancellation_reason"] == "universe_removed"
