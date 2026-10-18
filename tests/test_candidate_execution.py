"""Execution defects exposed by replaying long crypto histories with volume limits."""

import polars as pl
import pytest
from test_execution_v3 import bar

from backtest_engine.execution.models import OrderRequest
from backtest_engine.execution.simulator import ExecutionSimulatorV3
from backtest_engine.intraday.event.metrics import (
    filled_order_count,
    order_count,
    partial_fill_count,
)


@pytest.mark.parametrize(
    "step,volume,expected", [(0.1, 394.97, 394.9), (0.0001, 0.007496, 0.0074), (1, 16.87, 16)]
)
def test_fractional_partial_fills_respect_instrument_step(step, volume, expected):
    simulator = ExecutionSimulatorV3(volume_participation_limit=1, quantity_step=step)
    simulator.submit(OrderRequest("buy", "buy", volume * 2, time_in_force="ioc"))
    fills = simulator.process_bar(bar(1, volume=volume), bar_index=0)
    assert fills[0].quantity == pytest.approx(expected)
    assert fills[0].quantity <= volume
    assert simulator.orders["buy"].status == "expired"


def test_capacity_below_one_step_does_not_fill_and_does_not_consume_volume():
    simulator = ExecutionSimulatorV3(volume_participation_limit=1, quantity_step=0.1)
    simulator.submit(OrderRequest("buy", "buy", 0.2, time_in_force="gtc"))
    assert simulator.process_bar(bar(1, volume=0.09), bar_index=0) == []
    fills = simulator.process_bar(bar(2, volume=0.2), bar_index=1)
    assert fills[0].quantity == 0.2


def test_partial_expired_and_persistent_partial_are_counted_as_fills():
    events = pl.DataFrame(
        {
            "order_status": ["filled", "partial", "partial_expired", "partial_filled"],
            "requested_contract_delta": [1.0] * 4,
        }
    )
    assert order_count(events) == filled_order_count(events) == 4
    assert partial_fill_count(events) == 3
