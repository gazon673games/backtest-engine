"""Funding rates are cash-flow events, not bar attributes to forward-fill."""

from __future__ import annotations

from math import isfinite
from typing import Any

from backtest_engine.core.exceptions import DatasetValidationError


def funding_payment_for_row(
    row: dict[str, Any],
    *,
    quantity: float,
    point_value: float,
    mark_price: float,
    rate_column: str,
    default_rate: float,
    event_column: str,
    require_event_flag: bool,
) -> float:
    # The canonical price/rate sum also handles offsetting rates at different
    # prices. A custom rate column and legacy unpriced rows retain bar valuation.
    priced = rate_column == "funding_rate" and row.get("funding_price_rate") is not None
    rate = funding_rate_for_row(
        row,
        rate_column="funding_price_rate" if priced else rate_column,
        default_rate=0.0 if priced else default_rate,
        event_column=event_column,
        require_event_flag=require_event_flag,
    )
    unit_payment = rate if priced else mark_price * rate
    if not isfinite(unit_payment):
        raise DatasetValidationError("Funding payment requires a finite price/rate value")
    return quantity * point_value * unit_payment


def funding_rate_for_row(
    row: dict[str, Any],
    *,
    rate_column: str,
    default_rate: float,
    event_column: str,
    require_event_flag: bool,
) -> float:
    raw_rate = row.get(rate_column)
    rate = default_rate if raw_rate is None else float(raw_rate)
    if rate == 0.0:
        return 0.0
    raw_event = row.get(event_column)
    if raw_event is None:
        if require_event_flag:
            raise DatasetValidationError(
                f"Non-zero funding rate requires explicit {event_column}=true; "
                "forward-filled rates must not be charged on every bar"
            )
        return rate
    return rate if parse_event_flag(raw_event, event_column) else 0.0


def parse_event_flag(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise DatasetValidationError(f"{field} must be a boolean funding-event flag")
