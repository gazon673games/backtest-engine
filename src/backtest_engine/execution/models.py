from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from backtest_engine.core.exceptions import DatasetValidationError

ORDER_SIDES = frozenset({"buy", "sell"})
ORDER_STATUSES = frozenset(
    {"created", "active", "partially_filled", "filled", "cancelled", "expired", "rejected"}
)
ORDER_TYPES = frozenset({"market", "limit", "stop", "stop_limit", "trailing_stop"})
TIME_IN_FORCE = frozenset({"ioc", "fok", "day", "gtc", "bars"})
EXECUTION_DATA_MODES = frozenset({"bar", "trade", "quote"})
SELF_TRADE_PREVENTION = frozenset({"none", "expire_taker", "expire_maker", "expire_both"})


@dataclass(frozen=True)
class ExecutionBar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    bid: float | None = None
    ask: float | None = None
    market_reference_price: float | None = None


@dataclass(frozen=True)
class OrderRequest:
    order_id: str
    side: str
    quantity: float
    order_type: str = "market"
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "ioc"
    expires_after_bars: int | None = None
    oco_group: str | None = None
    parent_order_id: str | None = None
    trailing_offset: float | None = None
    created_ts: datetime | None = None
    post_only: bool = False
    reduce_only: bool = False
    self_trade_prevention: str = "none"


@dataclass(frozen=True)
class OrderState:
    request: OrderRequest
    status: str = "created"
    created_bar: int = 0
    updated_bar: int = 0
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    high_watermark: float | None = None
    low_watermark: float | None = None
    stop_triggered: bool = False
    rejection_reason: str | None = None

    @property
    def remaining_quantity(self) -> float:
        return max(self.request.quantity - self.filled_quantity, 0.0)

    @property
    def is_open(self) -> bool:
        return self.status in {"created", "active", "partially_filled"}


@dataclass(frozen=True)
class FillEvent:
    order_id: str
    ts: datetime
    side: str
    quantity: float
    price: float
    bar_index: int
    status: str
    reason: str
    liquidity_role: str = "taker"


def validate_order_request(request: OrderRequest) -> None:
    if not request.order_id.strip():
        raise DatasetValidationError("order_id must not be empty")
    if request.side not in ORDER_SIDES:
        raise DatasetValidationError(f"Unsupported order side: {request.side}")
    if request.order_type not in ORDER_TYPES:
        raise DatasetValidationError(f"Unsupported order type: {request.order_type}")
    if request.time_in_force not in TIME_IN_FORCE:
        raise DatasetValidationError(f"Unsupported time in force: {request.time_in_force}")
    if request.quantity <= 0.0:
        raise DatasetValidationError("order quantity must be positive")
    if request.order_type in {"limit", "stop_limit"} and request.limit_price is None:
        raise DatasetValidationError(f"{request.order_type} order requires limit_price")
    if request.order_type in {"stop", "stop_limit"} and request.stop_price is None:
        raise DatasetValidationError(f"{request.order_type} order requires stop_price")
    if request.order_type == "trailing_stop" and request.trailing_offset is None:
        raise DatasetValidationError("trailing_stop order requires trailing_offset")
    if request.trailing_offset is not None and request.trailing_offset <= 0.0:
        raise DatasetValidationError("trailing_offset must be positive")
    if request.post_only and request.order_type not in {"limit", "stop_limit"}:
        raise DatasetValidationError("post_only requires a limit or stop_limit order")
    if request.self_trade_prevention not in SELF_TRADE_PREVENTION:
        raise DatasetValidationError(
            "Unsupported self_trade_prevention: " + request.self_trade_prevention
        )
    if request.time_in_force == "bars":
        if request.expires_after_bars is None or request.expires_after_bars <= 0:
            raise DatasetValidationError("bars time_in_force requires expires_after_bars > 0")


def reduces_position(request: OrderRequest, position_quantity: float | None) -> bool:
    if position_quantity is None or position_quantity == 0.0:
        return False
    return (position_quantity > 0.0 and request.side == "sell") or (
        position_quantity < 0.0 and request.side == "buy"
    )


def parse_execution_bar(row: dict[str, Any], data_mode: str = "bar") -> ExecutionBar:
    ts = row["ts"]
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if not isinstance(ts, datetime):
        raise DatasetValidationError(f"ts must contain datetime values, got {ts!r}")
    if data_mode not in EXECUTION_DATA_MODES:
        raise DatasetValidationError(f"Unsupported execution data mode: {data_mode}")
    if data_mode == "trade":
        price = _required_positive_float(row.get("trade_price"), "trade_price")
        return ExecutionBar(
            ts=ts,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=_optional_float(row.get("trade_size")),
        )
    if data_mode == "quote":
        bid = _required_positive_float(row.get("bid"), "bid")
        ask = _required_positive_float(row.get("ask"), "ask")
        if bid > ask:
            raise DatasetValidationError(f"bid must not exceed ask: bid={bid}, ask={ask}")
        return ExecutionBar(
            ts=ts,
            open=(bid + ask) / 2.0,
            high=ask,
            low=bid,
            close=(bid + ask) / 2.0,
            volume=_optional_float(row.get("quote_size") or row.get("volume")),
            bid=bid,
            ask=ask,
        )
    return ExecutionBar(
        ts=ts,
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=_optional_float(row.get("volume")),
        bid=_optional_float(row.get("bid")),
        ask=_optional_float(row.get("ask")),
        market_reference_price=_optional_float(row.get("market_reference_price")),
    )


def fill_order_state(
    state: OrderState,
    *,
    quantity: float,
    price: float,
    bar_index: int,
) -> OrderState:
    total_quantity = state.filled_quantity + quantity
    if total_quantity <= 0.0:
        avg_price = 0.0
    else:
        avg_price = (
            state.avg_fill_price * state.filled_quantity + price * quantity
        ) / total_quantity
    status = "filled" if total_quantity >= state.request.quantity else "partially_filled"
    return replace(
        state,
        status=status,
        updated_bar=bar_index,
        filled_quantity=total_quantity,
        avg_fill_price=avg_price,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _required_positive_float(value: Any, field: str) -> float:
    if value is None:
        raise DatasetValidationError(f"{field} is required for execution data mode")
    parsed = float(value)
    if parsed <= 0.0:
        raise DatasetValidationError(f"{field} must be positive")
    return parsed
