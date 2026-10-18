from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.execution.models import OrderRequest
from backtest_engine.execution.order_book import OrderBookReplaySimulator
from backtest_engine.intraday.event.execution_l2 import IntradayL2Execution


def ts(seconds: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)


def row(
    sequence: int,
    side: str,
    price: float,
    size: float,
    *,
    seconds: int = 0,
    update_type: str = "snapshot",
    previous_sequence: int | None = None,
) -> dict[str, object]:
    return {
        "ts": ts(seconds),
        "sequence": sequence,
        "previous_sequence": previous_sequence,
        "update_type": update_type,
        "side": side,
        "price": price,
        "size": size,
    }


def snapshot_rows() -> list[dict[str, object]]:
    return [
        row(1, "bid", 99.0, 4.0),
        row(1, "ask", 101.0, 2.0),
        row(1, "ask", 102.0, 4.0),
    ]


def test_l2_market_sweeps_depth_and_reports_vwap() -> None:
    simulator = OrderBookReplaySimulator(snapshot_rows())
    simulator.process_until(ts(0))
    state = simulator.submit(
        OrderRequest(order_id="market", side="buy", quantity=3.0),
        ts=ts(0),
    )
    fills = simulator.process_until(ts(0))

    assert state.status == "filled"
    assert [fill.price for fill in fills] == [101.0, 102.0]
    assert simulator.orders["market"].avg_fill_price == pytest.approx(101 + 1 / 3)


def test_l2_post_only_fok_and_reduce_only_fail_closed() -> None:
    simulator = OrderBookReplaySimulator(snapshot_rows())
    simulator.process_until(ts(0))
    post_only = simulator.submit(
        OrderRequest(
            order_id="post",
            side="buy",
            quantity=1.0,
            order_type="limit",
            limit_price=101.0,
            time_in_force="gtc",
            post_only=True,
        ),
        ts=ts(0),
    )
    fok = simulator.submit(
        OrderRequest(
            order_id="fok",
            side="buy",
            quantity=3.0,
            order_type="limit",
            limit_price=101.0,
            time_in_force="fok",
        ),
        ts=ts(0),
    )
    reduce_only = simulator.submit(
        OrderRequest(
            order_id="reduce",
            side="sell",
            quantity=1.0,
            reduce_only=True,
        ),
        ts=ts(0),
        position_quantity=0.0,
    )

    assert post_only.status == "rejected"
    assert post_only.rejection_reason == "post_only_would_take"
    assert fok.status == "expired"
    assert reduce_only.rejection_reason == "reduce_only_rejected"


def test_l2_queue_and_sequence_gap_are_explicit() -> None:
    rows = snapshot_rows() + [
        row(
            2,
            "bid",
            99.0,
            1.0,
            seconds=1,
            update_type="delta",
            previous_sequence=1,
        )
    ]
    simulator = OrderBookReplaySimulator(rows, queue_ahead_fraction=0.5)
    simulator.process_until(ts(0))
    simulator.submit(
        OrderRequest(
            order_id="maker",
            side="buy",
            quantity=1.0,
            order_type="limit",
            limit_price=99.0,
            time_in_force="gtc",
            post_only=True,
        ),
        ts=ts(0),
    )
    fills = simulator.process_until(ts(1))
    assert fills[0].liquidity_role == "maker"
    assert fills[0].quantity == 1.0

    broken = OrderBookReplaySimulator(
        snapshot_rows() + [row(3, "bid", 99.0, 1.0, seconds=1, update_type="delta")]
    )
    with pytest.raises(DatasetValidationError, match="sequence gap"):
        broken.process_until(ts(1))


def test_l2_bridge_preserves_pending_fills_when_new_order_arrives(tmp_path: Path) -> None:
    rows = snapshot_rows() + [
        row(
            2,
            "bid",
            99.0,
            0.0,
            seconds=1,
            update_type="delta",
            previous_sequence=1,
        )
    ]
    path = tmp_path / "book.parquet"
    pl.DataFrame(rows).with_columns(pl.lit("book").alias("data_kind")).write_parquet(path)
    bridge = IntradayL2Execution(path, queue_ahead_fraction=0.0)
    bridge.submit(
        OrderRequest(
            order_id="maker",
            side="buy",
            quantity=1.0,
            order_type="limit",
            limit_price=99.0,
            time_in_force="gtc",
            post_only=True,
        ),
        decision_ts=ts(0),
        fill_deadline=ts(0),
        position_quantity=0.0,
    )

    _, fills = bridge.submit(
        OrderRequest(order_id="taker", side="buy", quantity=1.0),
        decision_ts=ts(1),
        fill_deadline=ts(1),
        position_quantity=0.0,
    )

    assert [fill.order_id for fill in fills] == ["maker", "taker"]
    assert bridge.simulator.position_quantity == 2.0


def test_reduce_only_fok_never_partially_fills() -> None:
    simulator = OrderBookReplaySimulator(snapshot_rows())
    simulator.process_until(ts(0))

    state = simulator.submit(
        OrderRequest(
            order_id="reduce-fok",
            side="sell",
            quantity=3.0,
            order_type="limit",
            limit_price=99.0,
            time_in_force="fok",
            reduce_only=True,
        ),
        ts=ts(0),
        position_quantity=2.0,
    )

    assert state.status == "expired"
    assert simulator.process_until(ts(0)) == []
    assert simulator.position_quantity == 2.0
