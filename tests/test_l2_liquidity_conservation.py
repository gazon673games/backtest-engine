"""Independent liquidity budgets and arrival priority for synthetic maker orders."""

from collections import defaultdict
from math import inf, nan

import pytest
from test_order_book_execution import row, ts

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.execution.models import OrderRequest, validate_order_request
from backtest_engine.execution.order_book import OrderBookReplaySimulator


def maker(order_id, side="buy", quantity=2.0):
    return OrderRequest(
        order_id=order_id,
        side=side,
        quantity=quantity,
        order_type="limit",
        limit_price=99.0 if side == "buy" else 101.0,
        time_in_force="gtc",
        post_only=True,
    )


@pytest.mark.parametrize("side", ["buy", "sell"])
@pytest.mark.parametrize("fraction,expected", [(0.0, [2.0, 2.0]), (0.5, [2.0]), (1.0, [])])
def test_one_book_decrease_has_one_shared_budget_and_arrival_priority(side, fraction, expected):
    book_side, price = ("bid", 99.0) if side == "buy" else ("ask", 101.0)
    rows = [row(1, "bid", 99.0, 4.0), row(1, "ask", 101.0, 4.0)]
    rows.append(row(2, book_side, price, 0.0, seconds=1, update_type="delta"))
    sim = OrderBookReplaySimulator(rows, queue_ahead_fraction=fraction)
    sim.process_until(ts(0))
    sim.submit(maker("z-first", side), ts=ts(0))
    sim.submit(maker("a-second", side, 3.0), ts=ts(0))
    fills = sim.process_until(ts(1))
    assert [fill.quantity for fill in fills] == expected
    assert [fill.order_id for fill in fills] == ["z-first", "a-second"][: len(expected)]
    assert sum(fill.quantity for fill in fills) <= 4.0 * (1.0 - fraction)


def test_default_queue_advances_once_and_later_trade_cannot_fill_both_orders():
    rows = [row(1, "bid", 99.0, 4.0), row(1, "ask", 101.0, 4.0)]
    for sequence, size in [(2, 0.0), (3, 2.0), (4, 1.0)]:
        rows.append(row(sequence, "bid", 99.0, size, seconds=sequence - 1, update_type="delta"))
    sim = OrderBookReplaySimulator(rows)
    sim.process_until(ts(0))
    sim.submit(maker("z-first", quantity=1.0), ts=ts(0))
    sim.submit(maker("a-second", quantity=1.0), ts=ts(0))
    assert sim.process_until(ts(1)) == []
    assert sim.process_until(ts(2)) == []
    fills = sim.process_until(ts(3))
    assert [(fill.order_id, fill.quantity) for fill in fills] == [("z-first", 1.0)]
    assert sim.orders["a-second"].filled_quantity == 0


def test_cancelling_front_order_releases_its_place_without_spending_liquidity():
    sim = OrderBookReplaySimulator(
        [
            row(1, "bid", 99.0, 2.0),
            row(1, "ask", 101.0, 2.0),
            row(2, "bid", 99.0, 1.0, seconds=1, update_type="delta"),
        ],
        queue_ahead_fraction=0.0,
    )
    sim.process_until(ts(0))
    sim.submit(maker("first"), ts=ts(0))
    sim.submit(maker("second"), ts=ts(0))
    sim.cancel("first")
    fills = sim.process_until(ts(1))
    assert [(fill.order_id, fill.quantity) for fill in fills] == [("second", 1.0)]


def test_bid_and_ask_decreases_have_independent_liquidity_budgets():
    sim = OrderBookReplaySimulator(
        [
            row(1, "bid", 99.0, 2.0),
            row(1, "ask", 101.0, 2.0),
            row(2, "bid", 99.0, 1.0, seconds=1, update_type="delta"),
            row(2, "ask", 101.0, 1.0, seconds=1, update_type="delta"),
        ],
        queue_ahead_fraction=0.0,
    )
    sim.process_until(ts(0))
    for side in ("buy", "sell"):
        for suffix in ("first", "second"):
            sim.submit(maker(side + suffix, side), ts=ts(0))
    quantities = defaultdict(float)
    for fill in sim.process_until(ts(1)):
        quantities[fill.side] += fill.quantity
    assert dict(quantities) == {"buy": 1.0, "sell": 1.0}


def test_reduce_only_resting_order_cannot_increase_a_reversed_position():
    sim = OrderBookReplaySimulator(
        [
            row(1, "bid", 99.0, 2.0),
            row(1, "ask", 101.0, 2.0),
            row(2, "ask", 101.0, 0.0, seconds=1, update_type="delta"),
        ],
        queue_ahead_fraction=0.0,
    )
    sim.process_until(ts(0))
    sim.submit(
        OrderRequest(
            order_id="reduce",
            side="sell",
            quantity=1.0,
            order_type="limit",
            limit_price=101.0,
            time_in_force="gtc",
            reduce_only=True,
        ),
        ts=ts(0),
        position_quantity=1.0,
    )
    sim.position_quantity = -1.0
    assert sim.process_until(ts(1)) == []
    assert sim.position_quantity == -1.0


@pytest.mark.parametrize("field", ["price", "size"])
@pytest.mark.parametrize("value", [nan, inf, -inf, "invalid"])
def test_invalid_book_numbers_cannot_reach_execution(field, value):
    bad = row(1, "bid", 99.0, 1.0)
    bad[field] = value
    sim = OrderBookReplaySimulator([bad, row(1, "ask", 101.0, 1.0)])
    with pytest.raises(DatasetValidationError, match="finite"):
        sim.process_until(ts(0))


@pytest.mark.parametrize("field", ["quantity", "limit_price", "stop_price", "trailing_offset"])
@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_nonfinite_order_values_are_rejected_before_any_fill(field, value):
    values = {"order_id": "invalid", "side": "buy", "quantity": 1.0, field: value}
    with pytest.raises(DatasetValidationError, match="finite"):
        validate_order_request(OrderRequest(**values))
