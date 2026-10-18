from __future__ import annotations

from dataclasses import replace
from math import floor, isfinite

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.execution.bar_fills import (
    advance_trailing_state,
    bar_fill_price,
    impact_price,
    initialize_trailing_state,
    ordered_open_order_ids,
    stop_limit_triggered,
)
from backtest_engine.execution.brackets import bracket_exit_orders as bracket_exit_orders
from backtest_engine.execution.model import EXECUTION_INTRABAR_AMBIGUITY_POLICIES
from backtest_engine.execution.models import (
    ExecutionBar,
    FillEvent,
    OrderRequest,
    OrderState,
    fill_order_state,
    parse_execution_bar,
    reduces_position,
    validate_order_request,
)


class ExecutionSimulatorV3:
    def __init__(
        self,
        *,
        volume_participation_limit: float | None = None,
        volume_to_quantity_multiplier: float = 1.0,
        latency_bars: int = 0,
        queue_ahead_quantity: float = 0.0,
        market_impact_bps: float = 0.0,
        data_mode: str = "bar",
        intrabar_ambiguity_policy: str = "conservative",
        allow_fractional_quantity: bool = True,
        quantity_step: float | None = None,
        partial_fill_policy: str = "allow",
    ) -> None:
        if volume_participation_limit is not None and not (0.0 < volume_participation_limit <= 1.0):
            raise DatasetValidationError("volume_participation_limit must be in (0, 1]")
        if not isfinite(volume_to_quantity_multiplier) or volume_to_quantity_multiplier <= 0.0:
            raise DatasetValidationError("volume_to_quantity_multiplier must be positive")
        if not isinstance(latency_bars, int) or latency_bars < 0:
            raise DatasetValidationError("latency_bars must be non-negative")
        if not isfinite(queue_ahead_quantity) or queue_ahead_quantity < 0.0:
            raise DatasetValidationError("queue_ahead_quantity must be non-negative")
        if not isfinite(market_impact_bps) or market_impact_bps < 0.0:
            raise DatasetValidationError("market_impact_bps must be non-negative")
        if intrabar_ambiguity_policy not in EXECUTION_INTRABAR_AMBIGUITY_POLICIES:
            raise DatasetValidationError(
                "Unsupported intrabar ambiguity policy: " + intrabar_ambiguity_policy
            )
        if partial_fill_policy not in {"allow", "reject"}:
            raise DatasetValidationError("partial_fill_policy must be allow or reject")
        if quantity_step is not None and (not isfinite(quantity_step) or quantity_step <= 0):
            raise DatasetValidationError("quantity_step must be positive and finite")
        self.volume_participation_limit = volume_participation_limit
        self.volume_to_quantity_multiplier = volume_to_quantity_multiplier
        self.latency_bars = latency_bars
        self.queue_ahead_quantity = queue_ahead_quantity
        self.market_impact_bps = market_impact_bps
        self.data_mode = data_mode
        self.intrabar_ambiguity_policy = intrabar_ambiguity_policy
        self.allow_fractional_quantity = allow_fractional_quantity
        self.quantity_step = quantity_step
        self.partial_fill_policy = partial_fill_policy
        self.orders: dict[str, OrderState] = {}
        self.open_order_ids: set[str] = set()

    def submit(
        self,
        request: OrderRequest,
        *,
        bar_index: int = 0,
        created_ts: object | None = None,
        position_quantity: float | None = None,
    ) -> OrderState:
        validate_order_request(request)
        if request.order_id in self.orders:
            raise DatasetValidationError(f"Duplicate order_id: {request.order_id}")
        if request.reduce_only and not reduces_position(request, position_quantity):
            state = OrderState(
                request=request,
                status="rejected",
                created_bar=bar_index,
                updated_bar=bar_index,
            )
            self.orders[request.order_id] = state
            return state
        if request.created_ts is None and created_ts is not None:
            request = replace(request, created_ts=created_ts)
        state = OrderState(
            request=request,
            status="active",
            created_bar=bar_index,
            updated_bar=bar_index,
        )
        self.orders[request.order_id] = state
        self.open_order_ids.add(request.order_id)
        return state

    def cancel(
        self,
        order_id: str,
        *,
        bar_index: int,
        reason: str = "cancelled",
    ) -> OrderState:
        state = self._require_order(order_id)
        if not state.is_open:
            return state
        cancelled = replace(state, status=reason, updated_bar=bar_index)
        self.orders[order_id] = cancelled
        self.open_order_ids.discard(order_id)
        return cancelled

    def replace_order(
        self,
        order_id: str,
        request: OrderRequest,
        *,
        bar_index: int,
    ) -> OrderState:
        old_state = self.cancel(order_id, bar_index=bar_index, reason="cancelled")
        new_request = replace(request, parent_order_id=request.parent_order_id or order_id)
        state = self.submit(new_request, bar_index=bar_index)
        self.orders[order_id] = replace(old_state, updated_bar=bar_index)
        return state

    def process_bar(self, row: dict[str, object], *, bar_index: int) -> list[FillEvent]:
        bar = parse_execution_bar(row, self.data_mode)
        available_quantity = self._available_quantity(bar)
        fills: list[FillEvent] = []
        for order_id in ordered_open_order_ids(
            self.orders,
            self.open_order_ids,
            self.intrabar_ambiguity_policy,
        ):
            if order_id not in self.open_order_ids:
                continue
            state = self.orders[order_id]
            if bar_index < state.created_bar + self.latency_bars:
                continue
            if self._expires_before_fill(state, bar, bar_index):
                self.orders[order_id] = replace(state, status="expired", updated_bar=bar_index)
                self.open_order_ids.discard(order_id)
                continue
            state = initialize_trailing_state(state, bar, bar_index)
            self.orders[order_id] = state
            stop_limit_previously_triggered = state.stop_triggered
            if (
                state.request.order_type == "stop_limit"
                and not state.stop_triggered
                and stop_limit_triggered(state.request, bar)
            ):
                state = replace(state, stop_triggered=True, updated_bar=bar_index)
                self.orders[order_id] = state
            price = bar_fill_price(
                state.request,
                bar,
                self.intrabar_ambiguity_policy,
                stop_limit_previously_triggered=stop_limit_previously_triggered,
            )
            if price is None:
                self.orders[order_id] = advance_trailing_state(state, bar, bar_index)
                self._expire_ioc_if_needed(state, bar_index)
                continue
            price = impact_price(state.request, price, self.market_impact_bps)
            fill_quantity = self._fill_quantity(state, available_quantity)
            if state.request.time_in_force == "fok" and fill_quantity < state.remaining_quantity:
                self.orders[order_id] = replace(state, status="expired", updated_bar=bar_index)
                self.open_order_ids.discard(order_id)
                continue
            if fill_quantity <= 0.0:
                self._expire_ioc_if_needed(state, bar_index)
                continue
            if available_quantity is not None:
                available_quantity = max(available_quantity - fill_quantity, 0.0)
            filled_state = fill_order_state(
                state,
                quantity=fill_quantity,
                price=price,
                bar_index=bar_index,
            )
            self.orders[order_id] = filled_state
            fills.append(
                FillEvent(
                    order_id=order_id,
                    ts=bar.ts,
                    side=state.request.side,
                    quantity=fill_quantity,
                    price=price,
                    bar_index=bar_index,
                    status=filled_state.status,
                    reason=state.request.order_type,
                )
            )
            if state.request.time_in_force == "ioc" and filled_state.status != "filled":
                self.orders[order_id] = replace(
                    filled_state,
                    status="expired",
                    updated_bar=bar_index,
                )
                self.open_order_ids.discard(order_id)
            if filled_state.status == "filled":
                self.open_order_ids.discard(order_id)
                self._cancel_oco_siblings(filled_state, bar_index)
        return fills

    def active_orders(self) -> list[OrderState]:
        return [self.orders[order_id] for order_id in self.open_order_ids]

    def _require_order(self, order_id: str) -> OrderState:
        if order_id not in self.orders:
            raise DatasetValidationError(f"Unknown order_id: {order_id}")
        return self.orders[order_id]

    def _available_quantity(self, bar: ExecutionBar) -> float | None:
        if self.volume_participation_limit is None and self.queue_ahead_quantity == 0.0:
            return None
        if bar.volume is None:
            return 0.0
        return (
            bar.volume
            * (self.volume_participation_limit or 1.0)
            * self.volume_to_quantity_multiplier
        )

    def _expires_before_fill(
        self,
        state: OrderState,
        bar: ExecutionBar,
        bar_index: int,
    ) -> bool:
        request = state.request
        if request.time_in_force == "day":
            return request.created_ts is not None and bar.ts.date() > request.created_ts.date()
        if request.time_in_force == "bars":
            return bar_index - state.created_bar >= int(request.expires_after_bars or 0)
        return False

    def _expire_ioc_if_needed(self, state: OrderState, bar_index: int) -> None:
        if state.request.time_in_force == "ioc" and state.remaining_quantity > 0.0:
            self.orders[state.request.order_id] = replace(
                state,
                status="expired",
                updated_bar=bar_index,
            )
            self.open_order_ids.discard(state.request.order_id)

    def _fill_quantity(self, state: OrderState, available_quantity: float | None) -> float:
        if available_quantity is None:
            return state.remaining_quantity
        queue_adjusted = max(available_quantity - self.queue_ahead_quantity, 0.0)
        step = self.quantity_step or (None if self.allow_fractional_quantity else 1.0)
        if step:
            queue_adjusted = floor(queue_adjusted / step + 1e-10) * step
        if self.partial_fill_policy == "reject" and queue_adjusted < state.remaining_quantity:
            return 0.0
        return min(state.remaining_quantity, queue_adjusted)

    def _cancel_oco_siblings(self, state: OrderState, bar_index: int) -> None:
        group = state.request.oco_group
        if group is None:
            return
        for order_id, sibling in self.orders.items():
            if order_id == state.request.order_id:
                continue
            if sibling.request.oco_group == group and sibling.is_open:
                self.orders[order_id] = replace(
                    sibling,
                    status="cancelled",
                    updated_bar=bar_index,
                )
                self.open_order_ids.discard(order_id)
