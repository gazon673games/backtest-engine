"""Order matching over validated historical L2 snapshots and deltas."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.execution.book_state import L2OrderBook
from backtest_engine.execution.models import (
    FillEvent,
    OrderRequest,
    OrderState,
    fill_order_state,
    reduces_position,
    validate_order_request,
)


class OrderBookReplaySimulator:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        decrease_is_trade: bool = True,
        queue_ahead_fraction: float = 1.0,
    ) -> None:
        if not 0.0 <= queue_ahead_fraction <= 1.0:
            raise DatasetValidationError("queue_ahead_fraction must be in [0, 1]")
        self.rows = sorted(rows, key=lambda row: (_timestamp(row["ts"]), int(row["sequence"])))
        self.book = L2OrderBook()
        self.decrease_is_trade = decrease_is_trade
        self.queue_ahead_fraction = queue_ahead_fraction
        self.cursor = 0
        self.orders: dict[str, OrderState] = {}
        self.open_order_ids: set[str] = set()
        self.queue_ahead: dict[str, float] = {}
        self.position_quantity = 0.0
        self._fills: list[FillEvent] = []

    def process_until(self, ts: datetime) -> list[FillEvent]:
        while self.cursor < len(self.rows) and _timestamp(self.rows[self.cursor]["ts"]) <= ts:
            batch = self._next_batch()
            removed = self.book.apply_batch(batch)
            if self.decrease_is_trade:
                self._fill_resting_orders(removed, _timestamp(batch[-1]["ts"]))
        fills, self._fills = self._fills, []
        return fills

    def submit(
        self,
        request: OrderRequest,
        *,
        ts: datetime,
        position_quantity: float | None = None,
    ) -> OrderState:
        validate_order_request(request)
        if request.order_id in self.orders:
            raise DatasetValidationError(f"Duplicate order_id: {request.order_id}")
        if position_quantity is not None:
            self.position_quantity = position_quantity
        if request.reduce_only and not reduces_position(request, self.position_quantity):
            return self._reject(request, "reduce_only_rejected")
        conflict = self._self_trade_conflicts(request)
        if conflict and request.self_trade_prevention in {"expire_taker", "expire_both"}:
            if request.self_trade_prevention == "expire_both":
                self._cancel_conflicts(conflict)
            return self._reject(request, "self_trade_prevention")
        if conflict and request.self_trade_prevention == "expire_maker":
            self._cancel_conflicts(conflict)
        if request.post_only and self.book.crosses(request):
            return self._reject(request, "post_only_would_take")
        state = OrderState(request=request, status="active")
        self.orders[request.order_id] = state
        if self.book.crosses(request):
            return self._execute_taker(state, ts)
        if request.time_in_force in {"ioc", "fok"}:
            return self._expire(state)
        price = float(request.limit_price)
        book_side = "bid" if request.side == "buy" else "ask"
        self.queue_ahead[request.order_id] = (
            self.book.level_size(book_side, price) * self.queue_ahead_fraction
        )
        self.open_order_ids.add(request.order_id)
        return state

    def cancel(self, order_id: str) -> OrderState:
        if order_id not in self.orders:
            raise DatasetValidationError(f"Unknown order_id: {order_id}")
        state = self.orders[order_id]
        if not state.is_open:
            return state
        cancelled = replace(state, status="cancelled")
        self.orders[order_id] = cancelled
        self.open_order_ids.discard(order_id)
        self.queue_ahead.pop(order_id, None)
        return cancelled

    def _execute_taker(self, state: OrderState, ts: datetime) -> OrderState:
        request = state.request
        if request.time_in_force == "fok" and request.reduce_only:
            if state.remaining_quantity > abs(self.position_quantity):
                return self._expire(state)
        limit = request.limit_price if request.order_type != "market" else None
        quantity = self._reducible_quantity(request, state.remaining_quantity)
        if request.time_in_force == "fok" and self.book.available(request.side, limit) < quantity:
            return self._expire(state)
        for filled, price in self.book.sweep(request.side, quantity, limit):
            state = self._record_fill(state, filled, price, ts, "taker")
        if state.remaining_quantity > 0.0:
            state = self._expire(state)
        return state

    def _fill_resting_orders(self, removed: dict[tuple[str, float], float], ts: datetime) -> None:
        filled_by_level: dict[tuple[str, float], float] = {}
        blocked_levels: set[tuple[str, float]] = set()
        # queue_ahead retains submission order; set iteration cannot define FIFO.
        for order_id in tuple(self.queue_ahead):
            if order_id not in self.open_order_ids:
                continue
            state = self.orders[order_id]
            if state.request.reduce_only and not reduces_position(
                state.request, self.position_quantity
            ):
                self.cancel(order_id)
                continue
            side = "bid" if state.request.side == "buy" else "ask"
            price = float(state.request.limit_price)
            level = (side, price)
            consumed = max(removed.get(level, 0.0) - filled_by_level.get(level, 0.0), 0.0)
            ahead = self.queue_ahead.get(order_id, 0.0)
            self.queue_ahead[order_id] = max(ahead - consumed, 0.0)
            available = max(consumed - ahead, 0.0)
            quantity = min(state.remaining_quantity, available)
            quantity = self._reducible_quantity(state.request, quantity)
            if quantity > 0.0 and level not in blocked_levels:
                filled_by_level[level] = filled_by_level.get(level, 0.0) + quantity
                state = self._record_fill(state, quantity, price, ts, "maker")
            if state.is_open:
                blocked_levels.add(level)

    def _record_fill(
        self, state: OrderState, quantity: float, price: float, ts: datetime, role: str
    ) -> OrderState:
        updated = fill_order_state(state, quantity=quantity, price=price, bar_index=0)
        self.orders[state.request.order_id] = updated
        signed = quantity if state.request.side == "buy" else -quantity
        self.position_quantity += signed
        self._fills.append(
            FillEvent(
                order_id=state.request.order_id,
                ts=ts,
                side=state.request.side,
                quantity=quantity,
                price=price,
                bar_index=0,
                status=updated.status,
                reason="l2_match",
                liquidity_role=role,
            )
        )
        if updated.status == "filled":
            self.open_order_ids.discard(state.request.order_id)
            self.queue_ahead.pop(state.request.order_id, None)
        return updated

    def _next_batch(self) -> list[dict[str, Any]]:
        first = self.rows[self.cursor]
        key = (_timestamp(first["ts"]), int(first["sequence"]))
        batch = []
        while self.cursor < len(self.rows):
            row = self.rows[self.cursor]
            if (_timestamp(row["ts"]), int(row["sequence"])) != key:
                break
            batch.append(row)
            self.cursor += 1
        return batch

    def _reducible_quantity(self, request: OrderRequest, quantity: float) -> float:
        if request.reduce_only and not reduces_position(request, self.position_quantity):
            return 0.0
        return min(quantity, abs(self.position_quantity)) if request.reduce_only else quantity

    def _reject(self, request: OrderRequest, reason: str) -> OrderState:
        state = OrderState(request=request, status="rejected", rejection_reason=reason)
        self.orders[request.order_id] = state
        return state

    def _expire(self, state: OrderState) -> OrderState:
        expired = replace(state, status="expired")
        self.orders[state.request.order_id] = expired
        self.open_order_ids.discard(state.request.order_id)
        return expired

    def _self_trade_conflicts(self, request: OrderRequest) -> list[str]:
        if not self.book.crosses(request):
            return []
        return [
            order_id
            for order_id in self.open_order_ids
            if self.orders[order_id].request.side != request.side
        ]

    def _cancel_conflicts(self, order_ids: list[str]) -> None:
        for order_id in order_ids:
            self.orders[order_id] = replace(self.orders[order_id], status="cancelled")
            self.open_order_ids.discard(order_id)
            self.queue_ahead.pop(order_id, None)


def _timestamp(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise DatasetValidationError(f"L2 ts must be datetime, got {value!r}")
    return value
