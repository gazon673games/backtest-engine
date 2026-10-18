"""Guard against charging roll slippage on every held futures bar."""

from datetime import datetime

import pytest
from test_intraday_event_support import build_events, write_intraday_dataset, write_intraday_plugin

from backtest_engine.accounting.futures import (
    FuturesExecutionCostSpec,
    FuturesInstrumentSpec,
    cost_breakdown,
)


@pytest.mark.parametrize("position", [10.0, -10.0])
def test_roll_slippage_requires_roll_notional(position):
    costs = cost_breakdown(
        contract_delta=0,
        turnover_notional_cash=0,
        current_contracts=position,
        roll_notional_cash=0,
        instrument=FuturesInstrumentSpec(tick_value=2),
        execution=FuturesExecutionCostSpec(roll_cost_bps=3, roll_slippage_ticks=2),
    )
    assert costs.total_cost_cash == 0


@pytest.mark.parametrize("position", [10.0, -10.0])
def test_contract_roll_charges_once_and_next_held_interval_is_free(tmp_path, position):
    dataset = write_intraday_dataset(
        tmp_path,
        timestamps=[datetime(2024, 1, 2, 13, minute) for minute in (30, 35, 40, 45)],
        closes=[100.0, 110.0, 111.0, 112.0],
        active_contracts=["CRH4", "CRM4", "CRM4", "CRM4"],
    )
    direction = 1 if position > 0 else -1
    strategy = write_intraday_plugin(
        tmp_path, f"def generate_signal(ctx):\n    return {direction}.0\n"
    )
    events = build_events(
        tmp_path,
        dataset,
        strategy,
        sizing_mode="fixed_contracts",
        fixed_contracts=abs(position),
        initial_capital=10_000,
        tick_value=2,
        roll_cost_bps=3,
        roll_slippage_ticks=2,
        roll_pnl_policy="exclude_gap",
    )
    assert events["contract_roll"].to_list() == [True, False, False]
    assert events["roll_cost_cash"].to_list() == pytest.approx([40.33, 0, 0])
    assert sum(events["gross_pnl"]) == pytest.approx(position * 2)
    assert events["equity_cash"][-1] == pytest.approx(10_000 + position * 2 - 40.33)
