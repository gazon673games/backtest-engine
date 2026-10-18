"""Bridges bar decision times to an independent historical L2 stream."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.execution.models import FillEvent, OrderRequest, OrderState
from backtest_engine.execution.order_book import OrderBookReplaySimulator


class IntradayL2Execution:
    def __init__(
        self,
        path: Path,
        *,
        latency_ms: int = 0,
        queue_ahead_fraction: float = 1.0,
    ) -> None:
        if not path.is_file():
            raise DatasetValidationError(f"order_book_path does not exist: {path}")
        try:
            frame = pl.read_parquet(path)
        except Exception as exc:
            raise DatasetValidationError(f"Failed to read L2 dataset: {path}") from exc
        required = {"ts", "sequence", "update_type", "side", "price", "size"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DatasetValidationError("L2 dataset is missing columns: " + ", ".join(missing))
        if "data_kind" in frame.columns:
            kinds = set(frame.get_column("data_kind").drop_nulls().to_list())
            if kinds != {"book"}:
                raise DatasetValidationError(
                    f"order_book_path must contain data_kind=book: {kinds}"
                )
        self.simulator = OrderBookReplaySimulator(
            frame.to_dicts(), queue_ahead_fraction=queue_ahead_fraction
        )
        self.latency = timedelta(milliseconds=latency_ms)

    def submit(
        self,
        request: OrderRequest,
        *,
        decision_ts: datetime,
        fill_deadline: datetime,
        position_quantity: float,
    ) -> tuple[OrderState, list[FillEvent]]:
        if request.order_type not in {"market", "limit"}:
            raise DatasetValidationError(
                "Historical L2 execution currently supports market and limit orders"
            )
        eligible_ts = decision_ts + self.latency
        fills = self.simulator.process_until(eligible_ts)
        position_after_advance = position_quantity + sum(
            fill.quantity if fill.side == "buy" else -fill.quantity for fill in fills
        )
        state = self.simulator.submit(
            request,
            ts=eligible_ts,
            position_quantity=position_after_advance,
        )
        fills.extend(self.simulator.process_until(eligible_ts))
        if fill_deadline > eligible_ts:
            fills.extend(self.simulator.process_until(fill_deadline))
        return state, fills

    def advance(self, ts: datetime) -> list[FillEvent]:
        return self.simulator.process_until(ts)

    def cancel(self, order_id: str) -> OrderState:
        return self.simulator.cancel(order_id)

    @property
    def orders(self) -> dict[str, OrderState]:
        return self.simulator.orders

    @property
    def open_order_ids(self) -> set[str]:
        return self.simulator.open_order_ids


def as_datetime(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise DatasetValidationError(f"Execution timestamp must be datetime, got {value!r}")
    return value
