from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import StrategyPluginError

SIGNAL_VALUES = frozenset({0.0, 1.0})
INTRADAY_EVENT_SIGNAL_VALUES = frozenset({-1.0, 0.0, 1.0})


@dataclass(frozen=True)
class IntradayEventPluginDecision:
    target: float
    contracts: float | None = None
    reason: str | None = None
    features: dict[str, Any] = field(default_factory=dict)
    order_type: str = "market"
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "ioc"
    expires_after_bars: int | None = None
    trailing_offset: float | None = None
    oco_group: str | None = None
    order_id: str | None = None
    cancel_order_id: str | None = None
    replace_order_id: str | None = None
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    post_only: bool = False
    reduce_only: bool = False
    self_trade_prevention: str = "none"
    skip_unaffordable_add: bool = False


def normalize_plugin_signal(raw_signal: Any, *, expected_length: int) -> list[float]:
    if isinstance(raw_signal, pl.Series):
        values = raw_signal.to_list()
    elif isinstance(raw_signal, list):
        values = raw_signal
    else:
        raise StrategyPluginError("generate_signals must return a polars.Series or list[float]")

    if len(values) != expected_length:
        msg = (
            f"generate_signals returned wrong length: expected {expected_length}, got {len(values)}"
        )
        raise StrategyPluginError(msg)

    signal: list[float] = []
    for index, value in enumerate(values):
        signal.append(_normalize_signal_value(value, index=index))
    return signal


def normalize_intraday_signal(raw_signal: Any) -> float:
    signal_value = _normalize_scalar_signal(raw_signal)
    if signal_value not in SIGNAL_VALUES:
        msg = (
            "generate_signal returned unsupported signal value: "
            f"{signal_value}. Allowed values are 0.0 and 1.0"
        )
        raise StrategyPluginError(msg)
    return signal_value


def normalize_intraday_event_signal(raw_signal: Any) -> float:
    return normalize_intraday_event_decision(raw_signal).target


def normalize_intraday_event_decision(raw_signal: Any) -> IntradayEventPluginDecision:
    if not isinstance(raw_signal, dict):
        return IntradayEventPluginDecision(
            target=_normalize_intraday_event_signal_value(raw_signal)
        )

    if "target" in raw_signal:
        raw_target = raw_signal["target"]
    elif "signal" in raw_signal:
        raw_target = raw_signal["signal"]
    elif "target_position" in raw_signal:
        raw_target = raw_signal["target_position"]
    else:
        raise StrategyPluginError(
            "generate_signal decision dict must include target, signal, or target_position"
        )
    features = raw_signal.get("features", raw_signal.get("debug", {}))
    if features is None:
        features = {}
    if not isinstance(features, dict):
        raise StrategyPluginError("generate_signal decision features must be a dict")
    if "features" in raw_signal and "debug" in raw_signal:
        debug = raw_signal["debug"]
        if not isinstance(debug, dict):
            raise StrategyPluginError("generate_signal decision debug must be a dict")
        features = {**features, "debug": debug}
    if "plots" in raw_signal:
        features = {**features, "plots": raw_signal["plots"]}
    if "logs" in raw_signal:
        features = {**features, "logs": raw_signal["logs"]}
    order_payload = raw_signal.get("order", {})
    if order_payload is None:
        order_payload = {}
    if not isinstance(order_payload, dict):
        raise StrategyPluginError("generate_signal decision order must be a dict")
    order_type = str(order_payload.get("type", raw_signal.get("order_type", "market"))).lower()
    if order_type not in {"market", "limit", "stop", "stop_limit", "trailing_stop"}:
        msg = f"generate_signal decision order type is unsupported: {order_type}"
        raise StrategyPluginError(msg)
    skip_add = raw_signal.get("skip_unaffordable_add", False)
    if not isinstance(skip_add, bool):
        raise StrategyPluginError("skip_unaffordable_add must be boolean")
    return IntradayEventPluginDecision(
        target=_normalize_intraday_event_signal_value(raw_target),
        contracts=_optional_contracts(raw_signal.get("contracts")),
        reason=None if raw_signal.get("reason") is None else str(raw_signal.get("reason")),
        features=features,
        skip_unaffordable_add=skip_add,
        order_type=order_type,
        limit_price=_optional_float(
            order_payload.get("limit_price", raw_signal.get("limit_price"))
        ),
        stop_price=_optional_float(order_payload.get("stop_price", raw_signal.get("stop_price"))),
        time_in_force=str(order_payload.get("time_in_force", "ioc")).lower(),
        expires_after_bars=_optional_int(
            order_payload.get("expires_after_bars", raw_signal.get("expires_after_bars"))
        ),
        trailing_offset=_optional_float(
            order_payload.get("trailing_offset", raw_signal.get("trailing_offset"))
        ),
        oco_group=_optional_string(order_payload.get("oco_group", raw_signal.get("oco_group"))),
        order_id=_optional_string(order_payload.get("order_id", raw_signal.get("order_id"))),
        cancel_order_id=_optional_string(
            order_payload.get("cancel_order_id", raw_signal.get("cancel_order_id"))
        ),
        replace_order_id=_optional_string(
            order_payload.get("replace_order_id", raw_signal.get("replace_order_id"))
        ),
        take_profit_price=_optional_float(
            order_payload.get("take_profit_price", raw_signal.get("take_profit_price"))
        ),
        stop_loss_price=_optional_float(
            order_payload.get("stop_loss_price", raw_signal.get("stop_loss_price"))
        ),
        post_only=bool(order_payload.get("post_only", raw_signal.get("post_only", False))),
        reduce_only=bool(order_payload.get("reduce_only", raw_signal.get("reduce_only", False))),
        self_trade_prevention=str(
            order_payload.get(
                "self_trade_prevention",
                raw_signal.get("self_trade_prevention", "none"),
            )
        ).lower(),
    )


def _normalize_signal_value(value: Any, *, index: int) -> float:
    if isinstance(value, bool) or value is None:
        raise StrategyPluginError(
            f"generate_signals returned invalid signal value at index {index}: {value!r}"
        )
    try:
        signal_value = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"generate_signals returned non-numeric value at index {index}: {value!r}"
        raise StrategyPluginError(msg) from exc
    if not math.isfinite(signal_value):
        raise StrategyPluginError(
            f"generate_signals returned non-finite value at index {index}: {value!r}"
        )
    if signal_value not in SIGNAL_VALUES:
        msg = (
            "generate_signals returned unsupported signal value at index "
            f"{index}: {signal_value}. Allowed values are 0.0 and 1.0"
        )
        raise StrategyPluginError(msg)
    return signal_value


def _normalize_scalar_signal(raw_signal: Any) -> float:
    if isinstance(raw_signal, bool) or raw_signal is None:
        raise StrategyPluginError(f"generate_signal returned invalid signal value: {raw_signal!r}")
    try:
        signal_value = float(raw_signal)
    except (TypeError, ValueError) as exc:
        raise StrategyPluginError(
            f"generate_signal returned non-numeric value: {raw_signal!r}"
        ) from exc
    if not math.isfinite(signal_value):
        raise StrategyPluginError(f"generate_signal returned non-finite value: {raw_signal!r}")
    return signal_value


def _normalize_intraday_event_signal_value(raw_signal: Any) -> float:
    signal_value = _normalize_scalar_signal(raw_signal)
    if signal_value not in INTRADAY_EVENT_SIGNAL_VALUES:
        msg = (
            "generate_signal returned unsupported signal value: "
            f"{signal_value}. Allowed values are -1.0, 0.0, and 1.0"
        )
        raise StrategyPluginError(msg)
    return signal_value


def _optional_float(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise StrategyPluginError(
            f"generate_signal decision order price is non-numeric: {raw_value!r}"
        ) from exc
    if not math.isfinite(value):
        raise StrategyPluginError(
            f"generate_signal decision order price is non-finite: {raw_value!r}"
        )
    return value


def _optional_contracts(raw_value: Any) -> float | None:
    value = _optional_float(raw_value)
    if value is not None and value < 0.0:
        raise StrategyPluginError("generate_signal decision contracts must be non-negative")
    return value


def _optional_int(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise StrategyPluginError(
            f"generate_signal decision order integer field is invalid: {raw_value!r}"
        ) from exc
    if value <= 0:
        raise StrategyPluginError("generate_signal decision order integer field must be positive")
    return value


def _optional_string(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    return value or None
