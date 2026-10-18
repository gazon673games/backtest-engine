from __future__ import annotations

import math
from typing import Any, Literal

Target = Literal[-1.0, 0.0, 1.0]


def feature_snapshot(**features: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key, value in features.items():
        if value is None:
            snapshot[key] = None
        elif isinstance(value, float):
            snapshot[key] = None if not math.isfinite(value) else value
        else:
            snapshot[key] = value
    return snapshot


def event_decision(
    target: Target,
    *,
    reason: str | None = None,
    features: dict[str, Any] | None = None,
    order: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "target": float(target),
        "reason": reason,
        "features": features or {},
        "order": order or {"type": "market"},
    }


def long_decision(
    *,
    reason: str | None = None,
    features: dict[str, Any] | None = None,
    order: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return event_decision(1.0, reason=reason, features=features, order=order)


def short_decision(
    *,
    reason: str | None = None,
    features: dict[str, Any] | None = None,
    order: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return event_decision(-1.0, reason=reason, features=features, order=order)


def flat_decision(
    *,
    reason: str | None = None,
    features: dict[str, Any] | None = None,
    order: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return event_decision(0.0, reason=reason, features=features, order=order)


def limit_order(limit_price: float) -> dict[str, Any]:
    return {"type": "limit", "limit_price": limit_price}


def stop_order(stop_price: float) -> dict[str, Any]:
    return {"type": "stop", "stop_price": stop_price}


def stop_limit_order(stop_price: float, limit_price: float) -> dict[str, Any]:
    return {"type": "stop_limit", "stop_price": stop_price, "limit_price": limit_price}


def cancel_order(order_id: str) -> dict[str, Any]:
    return {"type": "market", "cancel_order_id": order_id}


def replace_order(order_id: str, replacement: dict[str, Any]) -> dict[str, Any]:
    return {**replacement, "replace_order_id": order_id}


def bracket_order(
    *,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
    order_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "market"}
    if order_id is not None:
        payload["order_id"] = order_id
    if take_profit_price is not None:
        payload["take_profit_price"] = take_profit_price
    if stop_loss_price is not None:
        payload["stop_loss_price"] = stop_loss_price
    return payload
