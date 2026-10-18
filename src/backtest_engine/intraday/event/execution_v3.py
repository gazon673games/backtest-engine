from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.execution.model import execution_interval
from backtest_engine.execution.models import OrderRequest
from backtest_engine.execution.simulator import ExecutionSimulatorV3, bracket_exit_orders
from backtest_engine.intraday.event.config import IntradayEventExecutionConfig
from backtest_engine.intraday.event.execution_l2 import IntradayL2Execution, as_datetime
from backtest_engine.intraday.event.execution_v3_support import (
    IntradayOrderIntent,
    confirmed_fill_result,
    execution_bar_row,
    fill_result,
    stop_limit_ambiguous_unfilled,
)


class IntradayExecutionBookV3:
    def __init__(
        self, execution: IntradayEventExecutionConfig, *, allow_fractional_contracts: bool = True
    ) -> None:
        self.execution = execution
        self.simulator = ExecutionSimulatorV3(
            volume_participation_limit=execution.volume_participation_limit,
            volume_to_quantity_multiplier=execution.volume_to_contracts_multiplier,
            allow_fractional_quantity=allow_fractional_contracts,
            partial_fill_policy=execution.partial_fill_policy,
            latency_bars=execution.latency_bars,
            queue_ahead_quantity=execution.queue_ahead_contracts,
            market_impact_bps=execution.market_impact_bps,
            data_mode=execution.data_mode,
            intrabar_ambiguity_policy=execution.intrabar_ambiguity_policy,
        )
        self.l2 = (
            IntradayL2Execution(
                execution.order_book_path,
                latency_ms=execution.latency_ms,
                queue_ahead_fraction=execution.l2_queue_ahead_fraction,
            )
            if execution.order_book_path is not None
            else None
        )
        self.intents: dict[str, IntradayOrderIntent] = {}
        self.brackets: dict[str, tuple[float | None, float | None]] = {}
        self.bracket_order_ids: dict[str, list[str]] = {}
        self.feedback_order_ids: set[str] = set()
        self.sequence = 0

    @property
    def has_active_orders(self) -> bool:
        return bool(self._open_order_ids())

    def submit_and_process(
        self,
        *,
        rows: Sequence[dict[str, Any]],
        index: int,
        order_type: str,
        limit_price: float | None,
        stop_price: float | None,
        time_in_force: str,
        expires_after_bars: int | None,
        trailing_offset: float | None,
        oco_group: str | None,
        order_id: str | None,
        cancel_order_id: str | None,
        replace_order_id: str | None,
        take_profit_price: float | None,
        stop_loss_price: float | None,
        post_only: bool,
        reduce_only: bool,
        self_trade_prevention: str,
        previous_signal: float,
        previous_contracts: float,
        requested_signal: float,
        requested_contracts: float,
        base_fill_price: float,
    ) -> dict[str, Any]:
        if requested_contracts == previous_contracts:
            if cancel_order_id is not None:
                self._cancel(cancel_order_id, index)
                self.feedback_order_ids.add(cancel_order_id)
                return fill_result(False, "cancelled")
            return fill_result(
                filled=True,
                status="not_required",
                fill_price=base_fill_price,
                target_signal=requested_signal,
                target_contracts=requested_contracts,
            )
        ambiguous = stop_limit_ambiguous_unfilled(
            rows=rows,
            index=index,
            execution=self.execution,
            order_type=order_type,
            stop_price=stop_price,
            limit_price=limit_price,
            side="buy" if requested_contracts > previous_contracts else "sell",
        )
        if ambiguous:
            return fill_result(False, "ambiguous_unfilled")
        if cancel_order_id is not None:
            self._cancel(cancel_order_id, index)
            self.feedback_order_ids.add(cancel_order_id)
        if replace_order_id is not None:
            self._cancel(replace_order_id, index)
            self.feedback_order_ids.add(replace_order_id)
        order_id = order_id or self._next_order_id()
        self.intents[order_id] = IntradayOrderIntent(
            previous_signal=previous_signal,
            previous_contracts=previous_contracts,
            target_signal=requested_signal,
            target_contracts=requested_contracts,
        )
        if self.l2 is not None and (take_profit_price is not None or stop_loss_price is not None):
            raise DatasetValidationError(
                "L2 execution does not yet support synthetic bracket orders"
            )
        if take_profit_price is not None or stop_loss_price is not None:
            self.brackets[order_id] = (take_profit_price, stop_loss_price)
        request = OrderRequest(
            order_id=order_id,
            side="buy" if requested_contracts > previous_contracts else "sell",
            quantity=abs(requested_contracts - previous_contracts),
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            expires_after_bars=expires_after_bars,
            trailing_offset=trailing_offset,
            oco_group=oco_group,
            post_only=post_only,
            reduce_only=reduce_only,
            self_trade_prevention=self_trade_prevention,
        )
        if self.l2 is not None:
            interval = execution_interval(
                rows, index, self.execution.timing, self.execution.data_mode
            )
            state, fills = self.l2.submit(
                request,
                decision_ts=as_datetime(rows[index].get("available_at") or rows[index]["ts"]),
                fill_deadline=interval.fill_ts,
                position_quantity=previous_contracts,
            )
            self.feedback_order_ids.add(order_id)
            return self._l2_result(fills, state)
        self.simulator.submit(
            request,
            bar_index=index,
            created_ts=rows[index]["ts"],
            position_quantity=previous_contracts,
        )
        self.feedback_order_ids.add(order_id)
        return self.process_pending(rows=rows, index=index, base_fill_price=base_fill_price)

    def process_pending(
        self,
        *,
        rows: Sequence[dict[str, Any]],
        index: int,
        base_fill_price: float,
    ) -> dict[str, Any]:
        if self.l2 is not None:
            interval = execution_interval(
                rows, index, self.execution.timing, self.execution.data_mode
            )
            fills = self.l2.advance(interval.fill_ts)
            if not fills:
                status = "active" if self.l2.open_order_ids else "unfilled"
                return fill_result(False, status)
            return self._l2_result(fills, self.l2.orders[fills[0].order_id])
        before = {
            order_id: self.simulator.orders[order_id] for order_id in self.simulator.open_order_ids
        }
        fills = self.simulator.process_bar(
            execution_bar_row(rows, index, self.execution, base_fill_price),
            bar_index=index,
        )
        self.feedback_order_ids.update(
            order_id
            for order_id, previous in before.items()
            if self.simulator.orders[order_id] != previous
        )
        if not fills:
            return fill_result(False, "active" if self.simulator.open_order_ids else "unfilled")
        if len({fill.order_id for fill in fills}) > 1:
            raise DatasetValidationError(
                "Multiple independent orders filled in one bar; aggregate intrabar ordering "
                "cannot be reconstructed from OHLC data"
            )
        fill = fills[0]
        intent = self.intents.get(fill.order_id)
        if intent is None:
            return fill_result(False, "unknown_order")
        state = self.simulator.orders[fill.order_id]
        result = confirmed_fill_result(fills, state, intent)
        self._submit_bracket_if_needed(fill.order_id, index)
        return result

    def _next_order_id(self) -> str:
        self.sequence += 1
        return f"intraday-v3-{self.sequence}"

    def order_statuses(self) -> list[dict[str, Any]]:
        order_ids = self.feedback_order_ids | self._open_order_ids()
        orders = self.l2.orders if self.l2 is not None else self.simulator.orders
        self.feedback_order_ids.clear()
        return [
            {
                "order_id": state.request.order_id,
                "status": state.status,
                "filled_quantity": state.filled_quantity,
                "remaining_quantity": state.remaining_quantity,
                "avg_fill_price": state.avg_fill_price,
                "order_type": state.request.order_type,
                "side": state.request.side,
                "rejection_reason": state.rejection_reason,
            }
            for order_id in order_ids
            if (state := orders.get(order_id)) is not None
        ]

    def _l2_result(self, fills: list[Any], state: Any) -> dict[str, Any]:
        if not fills:
            return fill_result(False, state.status)
        if len({fill.order_id for fill in fills}) > 1:
            raise DatasetValidationError(
                "Multiple independent L2 orders filled before one strategy event"
            )
        fill = fills[0]
        state = self.l2.orders[fill.order_id]
        intent = self.intents[fill.order_id]
        return confirmed_fill_result(fills, state, intent)

    def _cancel(self, order_id: str, index: int) -> None:
        if self.l2 is not None:
            self.l2.cancel(order_id)
        else:
            self.simulator.cancel(order_id, bar_index=index)

    def _open_order_ids(self) -> set[str]:
        return self.l2.open_order_ids if self.l2 is not None else self.simulator.open_order_ids

    def _submit_bracket_if_needed(
        self,
        order_id: str,
        bar_index: int,
    ) -> None:
        if order_id not in self.brackets:
            return
        take_profit, stop_loss = self.brackets[order_id]
        parent = self.simulator.orders[order_id]
        for bracket_id in self.bracket_order_ids.pop(order_id, []):
            self._cancel(bracket_id, bar_index)
            self.feedback_order_ids.add(bracket_id)
        self.bracket_order_ids[order_id] = []
        intent = self.intents[order_id]
        fraction = parent.filled_quantity / parent.request.quantity
        position = (
            intent.previous_contracts
            + (intent.target_contracts - intent.previous_contracts) * fraction
        )
        if position == 0.0:
            self.brackets.pop(order_id)
            return
        quantity = abs(position)
        exit_side = "sell" if position > 0.0 else "buy"
        for request in bracket_exit_orders(
            parent_order_id=f"{order_id}:bracket:{self._next_order_id()}",
            side=exit_side,
            quantity=quantity,
            take_profit_price=take_profit,
            stop_loss_price=stop_loss,
        ):
            self.intents[request.order_id] = IntradayOrderIntent(
                previous_signal=intent.previous_signal
                + (intent.target_signal - intent.previous_signal) * fraction,
                previous_contracts=position,
                target_signal=0.0,
                target_contracts=0.0,
            )
            self.simulator.submit(request, bar_index=bar_index)
            self.feedback_order_ids.add(request.order_id)
            self.bracket_order_ids[order_id].append(request.order_id)
        if not parent.is_open:
            self.brackets.pop(order_id)
