"""Defines common execution timing and interval semantics for all runners."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.bar_availability import row_available_at, row_open_ts

EXECUTION_TIMINGS = frozenset(
    {
        "signal_close_to_same_close",
        "signal_close_to_next_bar_close",
        "signal_close_to_next_bar_open_to_close",
        "signal_close_to_next_bar_close_to_following_close",
    }
)
EXECUTION_ORDER_TYPES = frozenset({"market", "limit", "stop", "stop_limit"})
EXECUTION_INTRABAR_AMBIGUITY_POLICIES = frozenset(
    {"conservative", "optimistic", "stop_first", "limit_first"}
)


@dataclass(frozen=True)
class ExecutionInterval:
    fill_index: int
    fill_ts: datetime
    fill_price: float
    exit_ts: datetime
    exit_price: float
    exit_index: int


@dataclass(frozen=True)
class OrderFill:
    filled: bool
    status: str
    fill_price: float


def event_loop_stop(height: int, timing: str) -> int:
    if timing in {
        "signal_close_to_next_bar_close",
        "signal_close_to_next_bar_close_to_following_close",
    }:
        return max(height - 2, 0)
    return max(height - 1, 0)


def execution_interval(
    rows: Sequence[dict[str, Any]],
    index: int,
    timing: str,
    data_mode: str = "bar",
) -> ExecutionInterval:
    _validate_timing(timing)
    row = rows[index]
    next_row = rows[index + 1]
    if timing == "signal_close_to_same_close":
        return ExecutionInterval(
            fill_index=index,
            fill_ts=row_available_at(row),
            fill_price=execution_reference_price(row, data_mode, "close"),
            exit_ts=row_available_at(next_row),
            exit_price=execution_reference_price(next_row, data_mode, "close"),
            exit_index=index + 1,
        )
    if timing == "signal_close_to_next_bar_open_to_close":
        return ExecutionInterval(
            fill_index=index + 1,
            fill_ts=row_open_ts(next_row),
            fill_price=execution_reference_price(next_row, data_mode, "open"),
            exit_ts=row_available_at(next_row),
            exit_price=execution_reference_price(next_row, data_mode, "close"),
            exit_index=index + 1,
        )
    following_row = rows[index + 2]
    return ExecutionInterval(
        fill_index=index + 1,
        fill_ts=row_available_at(next_row),
        fill_price=execution_reference_price(next_row, data_mode, "close"),
        exit_ts=row_available_at(following_row),
        exit_price=execution_reference_price(following_row, data_mode, "close"),
        exit_index=index + 2,
    )


def execution_exit_row(
    rows: Sequence[dict[str, Any]],
    index: int,
    timing: str,
    data_mode: str = "bar",
) -> dict[str, Any]:
    return rows[execution_interval(rows, index, timing, data_mode).exit_index]


def resolve_order_fill(
    *,
    order_type: str,
    limit_price: float | None,
    stop_price: float | None,
    previous_signal: float,
    requested_signal: float,
    base_fill_price: float,
    row: dict[str, Any],
    next_row: dict[str, Any],
    timing: str,
    intrabar_ambiguity_policy: str,
) -> OrderFill:
    if requested_signal == previous_signal:
        return OrderFill(filled=True, status="not_required", fill_price=base_fill_price)
    if order_type not in EXECUTION_ORDER_TYPES:
        raise DatasetValidationError(f"Unsupported order type: {order_type}")
    if intrabar_ambiguity_policy not in EXECUTION_INTRABAR_AMBIGUITY_POLICIES:
        raise DatasetValidationError(
            "Unsupported intrabar ambiguity policy: " + intrabar_ambiguity_policy
        )
    if order_type == "market":
        return OrderFill(filled=True, status="filled", fill_price=base_fill_price)

    interval_row = row if timing == "signal_close_to_same_close" else next_row
    interval_high = float(interval_row["high"])
    interval_low = float(interval_row["low"])
    side = 1.0 if requested_signal > previous_signal else -1.0
    if order_type == "limit":
        if limit_price is None:
            raise DatasetValidationError("limit order requires limit_price")
        filled = interval_low <= limit_price if side > 0.0 else interval_high >= limit_price
        return OrderFill(
            filled=filled,
            status="filled" if filled else "unfilled",
            fill_price=limit_price if filled else base_fill_price,
        )
    if order_type == "stop_limit":
        if stop_price is None or limit_price is None:
            raise DatasetValidationError("stop_limit order requires stop_price and limit_price")
        stop_touched = interval_high >= stop_price if side > 0.0 else interval_low <= stop_price
        limit_touched = interval_low <= limit_price if side > 0.0 else interval_high >= limit_price
        if not stop_touched or not limit_touched:
            return OrderFill(filled=False, status="unfilled", fill_price=base_fill_price)
        filled = intrabar_ambiguity_policy in {"optimistic", "stop_first"}
        return OrderFill(
            filled=filled,
            status="filled" if filled else "ambiguous_unfilled",
            fill_price=limit_price if filled else base_fill_price,
        )

    if stop_price is None:
        raise DatasetValidationError("stop order requires stop_price")
    filled = interval_high >= stop_price if side > 0.0 else interval_low <= stop_price
    return OrderFill(
        filled=filled,
        status="filled" if filled else "unfilled",
        fill_price=stop_price if filled else base_fill_price,
    )


def execution_interval_volume(
    row: dict[str, Any],
    next_row: dict[str, Any],
    timing: str,
    data_mode: str = "bar",
) -> float:
    interval_row = row if timing == "signal_close_to_same_close" else next_row
    volume_column = "trade_size" if data_mode == "trade" else "volume"
    raw_volume = interval_row.get(volume_column, 0.0)
    if raw_volume is None:
        return 0.0
    try:
        volume = float(raw_volume)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(volume) or volume < 0.0:
        return 0.0
    return volume


def execution_reference_price(
    row: dict[str, Any],
    data_mode: str,
    bar_field: str = "close",
) -> float:
    if data_mode == "trade":
        value = row.get("trade_price")
        if value is None:
            raise DatasetValidationError("trade_price is required for data_mode=trade")
        return float(value)
    if data_mode == "quote":
        bid = row.get("bid")
        ask = row.get("ask")
        if bid is None or ask is None:
            raise DatasetValidationError("bid and ask are required for data_mode=quote")
        bid_value = float(bid)
        ask_value = float(ask)
        if bid_value <= 0.0 or ask_value <= 0.0 or bid_value > ask_value:
            raise DatasetValidationError("quote requires 0 < bid <= ask")
        return (bid_value + ask_value) / 2.0
    return float(row[bar_field])


def _validate_timing(timing: str) -> None:
    if timing not in EXECUTION_TIMINGS:
        raise DatasetValidationError(f"Unsupported execution timing: {timing}")
