from __future__ import annotations

from datetime import datetime

import pytest

from backtest_engine.execution.models import OrderRequest
from backtest_engine.execution.simulator import ExecutionSimulatorV3, bracket_exit_orders


def bar(
    day: int,
    *,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
    volume: float = 10.0,
    bid: float | None = None,
    ask: float | None = None,
) -> dict[str, object]:
    return {
        "ts": datetime(2024, 1, day, 10, 0),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "bid": bid,
        "ask": ask,
    }


def test_market_order_uses_bid_ask_when_available() -> None:
    simulator = ExecutionSimulatorV3()
    simulator.submit(OrderRequest(order_id="buy-1", side="buy", quantity=2.0, order_type="market"))

    fills = simulator.process_bar(bar(1, bid=99.9, ask=100.2), bar_index=0)

    assert len(fills) == 1
    assert fills[0].price == pytest.approx(100.2)
    assert simulator.orders["buy-1"].status == "filled"


def test_persistent_limit_order_partially_fills_across_bars() -> None:
    simulator = ExecutionSimulatorV3(volume_participation_limit=0.1)
    simulator.submit(
        OrderRequest(
            order_id="limit-1",
            side="buy",
            quantity=3.0,
            order_type="limit",
            limit_price=99.0,
            time_in_force="gtc",
        )
    )

    first = simulator.process_bar(bar(1, low=98.0, volume=10.0), bar_index=0)
    second = simulator.process_bar(bar(2, low=98.5, volume=20.0), bar_index=1)

    assert [fill.quantity for fill in first + second] == pytest.approx([1.0, 2.0])
    assert simulator.orders["limit-1"].status == "filled"
    assert simulator.orders["limit-1"].avg_fill_price == pytest.approx(99.0)


def test_ioc_partial_fill_expires_remainder() -> None:
    simulator = ExecutionSimulatorV3(volume_participation_limit=0.1)
    simulator.submit(OrderRequest(order_id="ioc-1", side="buy", quantity=3.0, order_type="market"))

    fills = simulator.process_bar(bar(1, volume=10.0), bar_index=0)

    assert fills[0].quantity == pytest.approx(1.0)
    assert simulator.orders["ioc-1"].filled_quantity == pytest.approx(1.0)
    assert simulator.orders["ioc-1"].status == "expired"
    assert simulator.active_orders() == []


def test_bracket_oco_cancels_sibling_after_take_profit_fill() -> None:
    simulator = ExecutionSimulatorV3()
    for request in bracket_exit_orders(
        parent_order_id="entry-1",
        side="sell",
        quantity=1.0,
        take_profit_price=110.0,
        stop_loss_price=95.0,
    ):
        simulator.submit(request)

    fills = simulator.process_bar(bar(1, high=111.0, low=100.0), bar_index=0)

    assert [fill.order_id for fill in fills] == ["entry-1:take_profit"]
    assert simulator.orders["entry-1:take_profit"].status == "filled"
    assert simulator.orders["entry-1:stop_loss"].status == "cancelled"


def test_trailing_stop_updates_watermark_and_fills() -> None:
    simulator = ExecutionSimulatorV3()
    simulator.submit(
        OrderRequest(
            order_id="trail-1",
            side="sell",
            quantity=1.0,
            order_type="trailing_stop",
            trailing_offset=2.0,
            time_in_force="gtc",
        )
    )

    simulator.process_bar(bar(1, high=105.0, low=104.0), bar_index=0)
    fills = simulator.process_bar(
        bar(2, open_=105.0, high=106.0, low=103.5),
        bar_index=1,
    )

    assert simulator.orders["trail-1"].request.stop_price == pytest.approx(104.0)
    assert fills == []
    fills = simulator.process_bar(
        bar(3, open_=103.0, high=104.0, low=102.0),
        bar_index=2,
    )
    assert fills[0].price == pytest.approx(103.0)
    assert simulator.orders["trail-1"].status == "filled"


def test_bars_time_in_force_expires_before_late_fill() -> None:
    simulator = ExecutionSimulatorV3()
    simulator.submit(
        OrderRequest(
            order_id="limit-expire",
            side="buy",
            quantity=1.0,
            order_type="limit",
            limit_price=90.0,
            time_in_force="bars",
            expires_after_bars=1,
        ),
        bar_index=0,
    )

    simulator.process_bar(bar(1, low=95.0), bar_index=0)
    fills = simulator.process_bar(bar(2, low=89.0), bar_index=1)

    assert fills == []
    assert simulator.orders["limit-expire"].status == "expired"


def test_day_time_in_force_uses_submit_timestamp() -> None:
    simulator = ExecutionSimulatorV3()
    simulator.submit(
        OrderRequest(
            order_id="day-1",
            side="buy",
            quantity=1.0,
            order_type="limit",
            limit_price=90.0,
            time_in_force="day",
        ),
        bar_index=0,
        created_ts=datetime(2024, 1, 1, 10, 0),
    )

    fills = simulator.process_bar(bar(2, low=89.0), bar_index=1)

    assert fills == []
    assert simulator.orders["day-1"].status == "expired"


def test_latency_queue_and_market_impact_change_fill() -> None:
    simulator = ExecutionSimulatorV3(
        latency_bars=1,
        volume_participation_limit=0.5,
        queue_ahead_quantity=2.0,
        market_impact_bps=10.0,
    )
    simulator.submit(
        OrderRequest(order_id="realism-1", side="buy", quantity=4.0, order_type="market"),
        bar_index=0,
    )

    first = simulator.process_bar(bar(1, open_=100.0, volume=10.0), bar_index=0)
    second = simulator.process_bar(bar(2, open_=100.0, volume=10.0), bar_index=1)

    assert first == []
    assert second[0].quantity == pytest.approx(3.0)
    assert second[0].price == pytest.approx(100.1)
    assert simulator.orders["realism-1"].status == "expired"
