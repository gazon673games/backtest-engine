from datetime import datetime
from decimal import Decimal

import pytest

from backtest_engine.execution.models import OrderRequest
from backtest_engine.execution.simulator import ExecutionSimulatorV3
from backtest_engine.intraday.event.execution_v3_support import (
    IntradayOrderIntent,
    confirmed_fill_result,
)


@pytest.mark.parametrize("direction,side", [(1, "buy"), (-1, "sell")])
def test_partial_fill_keeps_confirmed_whole_quantity_exact(direction, side):
    simulator = ExecutionSimulatorV3(
        volume_participation_limit=0.1, allow_fractional_quantity=False
    )
    simulator.submit(OrderRequest("partial", side, 45))
    fills = simulator.process_bar(
        {
            "ts": datetime(2026, 9, 26),
            "open": 100,
            "high": 100,
            "low": 100,
            "close": 100,
            "volume": 260,
        },
        bar_index=0,
    )
    assert [fill.quantity for fill in fills] == [26]
    result = confirmed_fill_result(
        fills, simulator.orders["partial"], IntradayOrderIntent(0, 0, direction, direction * 45)
    )
    assert result["target_contracts"] == direction * 26
    assert Decimal(str(result["target_contracts"])) % 1 == 0


def test_confirmation_uses_cumulative_fills_when_order_continues():
    simulator = ExecutionSimulatorV3(volume_participation_limit=0.1)
    simulator.submit(OrderRequest("persistent", "sell", 45, time_in_force="gtc"))
    bar = {
        "ts": datetime(2026, 9, 26),
        "open": 100,
        "high": 100,
        "low": 100,
        "close": 100,
        "volume": 260,
    }
    first = simulator.process_bar(bar, bar_index=0)
    second = simulator.process_bar(bar | {"volume": 100}, bar_index=1)
    assert [fill.quantity for fill in first + second] == [26, 10]
    result = confirmed_fill_result(
        second, simulator.orders["persistent"], IntradayOrderIntent(1, 10, -1, -35)
    )
    assert result["target_contracts"] == -26
