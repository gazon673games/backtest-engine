from __future__ import annotations

from datetime import datetime

import pytest

from backtest_engine.execution.model import execution_reference_price
from backtest_engine.execution.models import OrderRequest
from backtest_engine.execution.simulator import ExecutionSimulatorV3


def quote_row() -> dict[str, object]:
    return {
        "ts": datetime(2024, 1, 1),
        "bid": 99.0,
        "ask": 101.0,
        "quote_size": 5.0,
    }


@pytest.mark.parametrize(
    ("side", "expected"),
    [("buy", 101.0), ("sell", 99.0)],
)
def test_quote_market_orders_cross_the_spread(side: str, expected: float) -> None:
    simulator = ExecutionSimulatorV3(data_mode="quote")
    simulator.submit(OrderRequest(order_id=side, side=side, quantity=1.0))

    fill = simulator.process_bar(quote_row(), bar_index=0)[0]

    assert fill.price == expected


def test_trade_mode_uses_trade_price_and_size_for_participation() -> None:
    simulator = ExecutionSimulatorV3(
        data_mode="trade",
        volume_participation_limit=0.5,
    )
    simulator.submit(OrderRequest(order_id="trade", side="buy", quantity=10.0))

    fill = simulator.process_bar(
        {
            "ts": datetime(2024, 1, 1),
            "trade_price": 123.0,
            "trade_size": 8.0,
        },
        bar_index=0,
    )[0]

    assert fill.price == 123.0
    assert fill.quantity == 4.0
    assert fill.status == "partially_filled"


def test_quote_reference_price_is_midpoint() -> None:
    assert execution_reference_price(quote_row(), "quote") == 100.0
