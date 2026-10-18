from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from backtest_engine.execution.models import OrderRequest, OrderState
from backtest_engine.execution.simulator import ExecutionSimulatorV3
from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig


@dataclass(frozen=True)
class MultiInstrumentFillResult:
    fill_price: float
    target_contracts: float
    filled_contract_delta: float
    fill_fraction: float
    order_status: str


class LegExecutionBook:
    """One persistent market order per leg; a changed target replaces it."""

    def __init__(self, config: MultiInstrumentBacktestConfig) -> None:
        self.simulator = ExecutionSimulatorV3(
            volume_participation_limit=config.execution.volume_participation_limit,
            volume_to_quantity_multiplier=config.execution.volume_to_contracts_multiplier,
            latency_bars=config.execution.latency_bars,
            queue_ahead_quantity=config.execution.queue_ahead_contracts,
            market_impact_bps=config.execution.market_impact_bps,
            allow_fractional_quantity=config.portfolio.allow_fractional_contracts,
            partial_fill_policy=config.execution.partial_fill_policy,
            data_mode=config.execution.data_mode,
        )
        self.requested_contracts = 0.0
        self.requested_target = 0.0
        self.before_fill: dict[str, OrderState] | None = None
        self.cancellations: dict[str, str] = {}
        self.symbol: str | None = None

    def cancel_pending(self, index: int, reason: str) -> None:
        for order_id in tuple(self.simulator.open_order_ids):
            self.simulator.cancel(order_id, bar_index=index)
            self.cancellations[order_id] = reason

    def update_instrument(self, symbol: str, index: int, position: float, target: float) -> None:
        if self.symbol is not None and self.symbol != symbol:
            self.cancel_pending(index, "contract_changed")
            self.requested_contracts, self.requested_target = position, target
        self.symbol = symbol

    def request(
        self, contracts: float, target: float, previous: float, index: int, ts: Any
    ) -> None:
        if contracts != self.requested_contracts:
            self.cancel_pending(index, "target_replaced")
        self.requested_contracts, self.requested_target = contracts, target
        delta = contracts - previous
        if delta and not self.simulator.open_order_ids:
            self.simulator.submit(
                OrderRequest(
                    order_id=f"{index}:{len(self.simulator.orders)}:market",
                    side="buy" if delta > 0 else "sell",
                    quantity=abs(delta),
                    order_type="market",
                    time_in_force="gtc",
                ),
                bar_index=index,
                created_ts=ts,
            )

    def rollback_fill(self) -> None:
        if self.before_fill is not None:
            self.simulator.orders.update(self.before_fill)
            self.simulator.open_order_ids = set(self.before_fill)

    def finish(self) -> int:
        pending = len(self.simulator.open_order_ids)
        for order_id in tuple(self.simulator.open_order_ids):
            state = self.simulator.orders[order_id]
            self.simulator.cancel(order_id, bar_index=state.updated_bar)
            self.cancellations[order_id] = "end_of_data"
        return pending

    def history(self, asset_id: str) -> list[dict[str, Any]]:
        return [
            {
                "asset_id": asset_id,
                "order_id": key,
                "side": state.request.side,
                "quantity": state.request.quantity,
                "filled_quantity": state.filled_quantity,
                "avg_fill_price": state.avg_fill_price,
                "status": state.status,
                "created_bar": state.created_bar,
                "updated_bar": state.updated_bar,
                "created_ts": state.request.created_ts,
                "cancellation_reason": self.cancellations.get(key),
            }
            for key, state in self.simulator.orders.items()
        ]


def market_fill_result(
    *,
    rows: Sequence[dict[str, Any]],
    index: int,
    base_fill_price: float,
    previous_contracts: float,
    requested_contracts: float,
    config: MultiInstrumentBacktestConfig,
    execution_index: int | None = None,
    book: LegExecutionBook | None = None,
) -> MultiInstrumentFillResult:
    if book is not None:
        book.before_fill = None
    contract_delta = requested_contracts - previous_contracts
    if config.execution.engine != "v3" or contract_delta == 0.0:
        return MultiInstrumentFillResult(
            fill_price=base_fill_price,
            target_contracts=requested_contracts,
            filled_contract_delta=contract_delta,
            fill_fraction=1.0,
            order_status="not_required" if contract_delta == 0.0 else "filled",
        )
    fill_index = index if execution_index is None else execution_index
    if book is None:
        book = LegExecutionBook(config)
        book.request(
            requested_contracts, 0.0, previous_contracts, fill_index, rows[fill_index]["ts"]
        )
    simulator = book.simulator
    book.before_fill = {key: simulator.orders[key] for key in simulator.open_order_ids}
    fills = simulator.process_bar(
        {
            "ts": rows[fill_index]["ts"],
            "open": rows[fill_index]["open"],
            "high": rows[fill_index]["high"],
            "low": rows[fill_index]["low"],
            "close": rows[fill_index]["close"],
            "market_reference_price": base_fill_price,
            "volume": rows[fill_index].get("volume"),
            "bid": rows[fill_index].get("bid"),
            "ask": rows[fill_index].get("ask"),
            "trade_price": rows[fill_index].get("trade_price"),
            "trade_size": rows[fill_index].get("trade_size"),
            "quote_size": rows[fill_index].get("quote_size"),
        },
        bar_index=fill_index,
    )
    if not fills:
        return MultiInstrumentFillResult(base_fill_price, previous_contracts, 0.0, 0.0, "unfilled")
    fill = fills[0]
    signed_fill = fill.quantity if contract_delta > 0.0 else -fill.quantity
    target_contracts = previous_contracts + signed_fill
    fill_fraction = 0.0 if contract_delta == 0.0 else abs(signed_fill / contract_delta)
    return MultiInstrumentFillResult(
        fill_price=fill.price,
        target_contracts=target_contracts,
        filled_contract_delta=signed_fill,
        fill_fraction=fill_fraction,
        order_status=fill.status,
    )
