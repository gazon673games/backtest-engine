from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from backtest_engine.execution.model import execution_interval
from backtest_engine.intraday.event.config import IntradayEventExecutionConfig


@dataclass(frozen=True)
class IntradayOrderIntent:
    previous_signal: float
    previous_contracts: float
    target_signal: float
    target_contracts: float


def confirmed_fill_result(
    fills: list[Any], state: Any, intent: IntradayOrderIntent
) -> dict[str, Any]:
    quantity = sum(fill.quantity for fill in fills)
    price = (
        fills[0].price
        if len(fills) == 1
        else sum(fill.quantity * fill.price for fill in fills) / quantity
    )
    fraction = min(state.filled_quantity / state.request.quantity, 1.0)
    # Reconstruct quantity from confirmed fills, not a ratio that introduces
    # off-step floating residues even for whole-contract executions.
    filled_delta = state.filled_quantity if state.request.side == "buy" else -state.filled_quantity
    result = fill_result(
        True,
        "partial_expired" if state.status == "expired" else state.status,
        price,
        intent.previous_signal + (intent.target_signal - intent.previous_signal) * fraction,
        intent.previous_contracts + filled_delta,
    )
    result["requested_contracts"] = intent.target_contracts
    return result


def execution_bar_row(
    rows: Sequence[dict[str, Any]],
    index: int,
    execution: IntradayEventExecutionConfig,
    base_fill_price: float,
) -> dict[str, Any]:
    interval = execution_interval(rows, index, execution.timing, execution.data_mode)
    interval_row = rows[interval.fill_index]
    return {
        "ts": interval.fill_ts,
        "open": interval_row["open"],
        "high": interval_row["high"],
        "low": interval_row["low"],
        "close": interval_row["close"],
        "market_reference_price": base_fill_price,
        "volume": interval_row.get("volume"),
        "bid": interval_row.get("bid"),
        "ask": interval_row.get("ask"),
        "trade_price": interval_row.get("trade_price"),
        "trade_size": interval_row.get("trade_size"),
        "quote_size": interval_row.get("quote_size"),
    }


def stop_limit_ambiguous_unfilled(
    *,
    rows: Sequence[dict[str, Any]],
    index: int,
    execution: IntradayEventExecutionConfig,
    order_type: str,
    stop_price: float | None,
    limit_price: float | None,
    side: str,
) -> bool:
    if order_type != "stop_limit" or stop_price is None or limit_price is None:
        return False
    if execution.intrabar_ambiguity_policy in {"optimistic", "stop_first"}:
        return False
    interval = execution_interval(rows, index, execution.timing, execution.data_mode)
    row = rows[interval.fill_index]
    stop_touched = row["high"] >= stop_price if side == "buy" else row["low"] <= stop_price
    limit_touched = row["low"] <= limit_price if side == "buy" else row["high"] >= limit_price
    return bool(stop_touched and limit_touched)


def fill_result(
    filled: bool,
    status: str,
    fill_price: float | None = None,
    target_signal: float | None = None,
    target_contracts: float | None = None,
) -> dict[str, Any]:
    return {
        "filled": filled,
        "status": status,
        "fill_price": fill_price,
        "target_signal": target_signal,
        "target_contracts": target_contracts,
    }
