from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from test_intraday_event_support import make_config, write_intraday_dataset

from backtest_engine.core.exceptions import DatasetValidationError, StrategyPluginError
from backtest_engine.intraday.event.add_affordability import skip_unaffordable_add
from backtest_engine.intraday.event.config import IntradayEventInstrumentSpec
from backtest_engine.intraday.event.loop import build_intraday_event_events
from backtest_engine.intraday.event.results import build_intraday_event_decision_actions
from backtest_engine.strategies.plugins.normalization import (
    IntradayEventPluginDecision,
    normalize_intraday_event_decision,
)


def check_add(
    tmp_path,
    *,
    side=1,
    previous=136,
    requested=168,
    capital=208532.4013,
    margin=1291.42,
    commission=0.5,
    max_margin_usage=None,
    enabled=True,
):
    config = make_config(
        tmp_path,
        tmp_path / "bars.parquet",
        tmp_path / "strategy.js",
        sizing_mode="strategy_contracts",
        initial_capital=capital,
        commission_bps=commission,
        slippage_ticks=1,
        bid_ask_spread_ticks=1,
        max_margin_usage=max_margin_usage,
    )
    decision = IntradayEventPluginDecision(
        target=side, contracts=requested, skip_unaffordable_add=enabled, reason="add"
    )
    state = {}
    skipped = skip_unaffordable_add(
        decision,
        previous_contracts=previous * side,
        equity_cash=capital,
        fill_price=8.988,
        fill_row={
            "ts": datetime(2022, 12, 6, 14, 45),
            "open": 8.988,
            "high": 8.99,
            "low": 8.987,
            "close": 8.988,
            "volume": 10000,
        },
        instrument=IntradayEventInstrumentSpec(
            symbol="CRZ2",
            price_currency="RUB",
            pnl_currency="RUB",
            contract_multiplier=1000,
            tick_size=0.001,
            tick_value=1,
            initial_margin=margin,
        ),
        config=config,
        decision_state=state,
    )
    return skipped, state


@pytest.mark.parametrize("side", [-1, 1])
def test_full_unaffordable_add_is_skipped_without_clipping(tmp_path, side):
    skipped, state = check_add(tmp_path, side=side)
    assert skipped
    assert state["requested_contracts"] == side * 168
    assert state["order_filled"] is False
    details = state["strategy_features"]["addAffordability"]
    assert details["requiredMargin"] == pytest.approx(216958.56)
    assert details["estimatedAddCostCash"] == pytest.approx(62.3808)
    assert details["previousContracts"] == side * 136


def test_execution_costs_and_margin_limit_are_reserved(tmp_path):
    assert check_add(tmp_path, capital=216970)[0]  # Margin fits, costs do not.
    assert not check_add(tmp_path, capital=217021)[0]
    assert check_add(tmp_path, capital=220000, max_margin_usage=0.95)[0]


@pytest.mark.parametrize("previous,requested", [(0, 168), (136, 40), (136, 0), (136, 136)])
def test_entries_reductions_and_closes_keep_existing_risk_behavior(tmp_path, previous, requested):
    assert not check_add(tmp_path, previous=previous, requested=requested, capital=1)[0]


def test_default_policy_does_not_silently_skip(tmp_path):
    assert not check_add(tmp_path, enabled=False)[0]


def test_unknown_margin_fails_explicitly(tmp_path):
    with pytest.raises(DatasetValidationError, match="requires initial_margin for CRZ2"):
        check_add(tmp_path, margin=None)


@pytest.mark.parametrize("value", ["false", 1, None])
def test_policy_requires_a_boolean(value):
    with pytest.raises(StrategyPluginError, match="skip_unaffordable_add must be boolean"):
        normalize_intraday_event_decision({"target": 1, "skip_unaffordable_add": value})


def run_fixture(tmp_path: Path, *, side=1, skip=True):
    # The skipped add must not receive the subsequent 400-point gain.
    prices = [100, 100, 100, 100, 500, 500, 500, 500, 500]
    if side < 0:
        prices = [600 - price for price in prices]
    path = write_intraday_dataset(
        tmp_path,
        timestamps=[datetime(2024, 1, 2, 10) + timedelta(minutes=5 * i) for i in range(9)],
        closes=prices,
    )
    plugin = tmp_path / "add.py"
    plugin.write_text(
        f"""
def initialize(params):
    return {{"quantity": 0, "calls": 0}}

def on_fill(state, event):
    state["quantity"] = event["target_contracts"]

def generate_signal(ctx, params):
    ctx.state["calls"] += 1
    count = ctx.state["calls"]
    if count == 7:
        return {{"target": 0, "contracts": 0, "skip_unaffordable_add": {skip}}}
    if count == 6:
        return {{"target": {-side}, "contracts": 40, "skip_unaffordable_add": {skip}}}
    size = 40 if count == 1 else abs(ctx.state["quantity"]) + 32
    return {{"target": {side}, "contracts": size, "skip_unaffordable_add": {skip}}}
""",
        encoding="utf-8",
    )
    config = make_config(
        tmp_path,
        path,
        plugin,
        execution_timing="signal_close_to_next_bar_close",
        initial_capital=110000,
        sizing_mode="strategy_contracts",
        initial_margin=1291.42,
        commission_bps=1,
        allow_fractional_contracts=False,
    )
    snapshots = []
    frame = pl.read_parquet(path)
    events = build_intraday_event_events(
        config,
        frame,
        {"timeframe": "5m", "bar_minutes": 5},
        ledger_snapshots=snapshots,
    )
    return events, snapshots


@pytest.mark.parametrize("side", [-1, 1])
def test_skipped_add_keeps_position_then_retries_and_accounting_reconciles(tmp_path, side):
    events, snapshots = run_fixture(tmp_path, side=side)
    assert events["target_contracts"].to_list() == [
        side * q for q in [40, 72, 72, 104, 104, -40, 0]
    ]
    skipped = events.filter(pl.col("strategy_reason") == "insufficient_margin_add_skipped")
    assert skipped["requested_contracts"].to_list() == [side * 104, side * 136]
    assert skipped["contract_delta"].to_list() == [0, 0]
    assert skipped["cost_cash"].to_list() == [0, 0]
    assert skipped["fill_fraction"].to_list() == [0, 0]
    assert skipped["order_filled"].to_list() == [False, False]
    # Only the existing 72 contracts earn the price move, not the rejected 104.
    assert events["gross_pnl"].sum() == pytest.approx(28800)
    expected_fees = sum(
        abs(row["contract_delta"]) * row["fill_price"] * 0.0001
        for row in events.iter_rows(named=True)
    )
    assert events["cost_cash"].sum() == pytest.approx(expected_fees)
    assert events["equity_cash"][-1] == pytest.approx(110000 + 28800 - expected_fees)
    assert snapshots[-1]["equity_cash"] == pytest.approx(events["equity_cash"][-1])
    actions = build_intraday_event_decision_actions(events)
    assert actions["reason"][2] == "insufficient_margin_add_skipped"


def test_unopted_strategy_still_fails_margin_invariant(tmp_path):
    with pytest.raises(DatasetValidationError, match="Post-fill risk breach"):
        run_fixture(tmp_path, skip=False)
