"""Shared market-price and margin inputs for every perpetual runtime."""

from math import isfinite
from typing import Any

from backtest_engine.accounting.contract_support import validate_accounting_contract
from backtest_engine.accounting.perpetual import MarginTier, TieredMarginSchedule
from backtest_engine.core.exceptions import DatasetValidationError


def market_value(row: dict[str, Any], column: str, fallback: Any, *, required: bool) -> float:
    value = row.get(column)
    if value is None:
        if required:
            raise DatasetValidationError(f"Strict perpetual accounting requires {column}")
        value = fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(f"{column} must be finite and positive") from exc
    if not isfinite(parsed) or parsed <= 0.0:
        raise DatasetValidationError(f"{column} must be finite and positive")
    return parsed


def perpetual_mark_prices(
    row: dict[str, Any],
    fallback: float,
    *,
    require_mark_price: bool,
    mark_price_column: str = "mark_price",
    mark_low_column: str = "mark_low",
    mark_high_column: str = "mark_high",
) -> tuple[float, float, float]:
    mark = market_value(row, mark_price_column, fallback, required=require_mark_price)
    low = market_value(row, mark_low_column, row.get("low", mark), required=require_mark_price)
    high = market_value(row, mark_high_column, row.get("high", mark), required=require_mark_price)
    if not low <= mark <= high:
        raise DatasetValidationError(
            "Perpetual mark range requires mark_low <= mark_price <= mark_high"
        )
    return mark, low, high


def perpetual_margin_schedule(
    instrument: Any, *, require_margin_tiers: bool
) -> TieredMarginSchedule:
    validate_accounting_contract(instrument)
    if not instrument.margin_tiers:
        if require_margin_tiers:
            raise DatasetValidationError("Strict perpetual accounting requires margin_tiers")
        tiers = [MarginTier(0.0, None, instrument.maintenance_margin_rate or 0.0)]
    else:
        tiers = [
            MarginTier(
                t.notional_floor, t.notional_cap, t.maintenance_margin_rate, t.maintenance_amount
            )
            for t in instrument.margin_tiers
        ]
    return TieredMarginSchedule(tiers)
