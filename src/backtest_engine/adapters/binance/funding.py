"""Retain the price associated with each Binance funding charge."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from backtest_engine.core.exceptions import DatasetValidationError


def funding_price_rate(events: list[dict[str, Any]], symbol: str) -> float | None:
    """Sum mark-price × rate; None explicitly retains legacy unpriced events."""
    total = Decimal(0)
    incomplete = False
    for event in events:
        try:
            rate = Decimal(str(event["fundingRate"]))
            raw_price = event.get("markPrice")
            price = None if raw_price in (None, "") else Decimal(str(raw_price))
        except (InvalidOperation, ValueError, KeyError) as error:
            raise DatasetValidationError(
                f"Invalid Binance funding price/rate for {symbol} at {event.get('fundingTime')}"
            ) from error
        if not rate.is_finite() or (price is not None and (not price.is_finite() or price <= 0)):
            raise DatasetValidationError(
                f"Invalid Binance funding price/rate for {symbol} at {event.get('fundingTime')}"
            )
        if price is None and rate != 0:
            incomplete = True
        elif price is not None:
            total += price * rate
    return None if incomplete else float(total)


def ordered_funding_events(events: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    """Do not double-charge identical observations or accept conflicting ones."""
    indexed = {}
    for event in events:
        funding_price_rate([event], symbol)
        try:
            timestamp = int(event["fundingTime"])
            if timestamp < 0 or str(timestamp) != str(event["fundingTime"]):
                raise ValueError("invalid timestamp")
        except (KeyError, ValueError, TypeError) as error:
            raise DatasetValidationError(
                f"Invalid Binance funding timestamp for {symbol}"
            ) from error
        if event.get("symbol", symbol) != symbol:
            raise DatasetValidationError(f"Binance funding symbol mismatch: expected {symbol}")
        previous = indexed.get(timestamp)
        if previous is not None and previous != event:
            raise DatasetValidationError(f"Conflicting Binance funding for {symbol} at {timestamp}")
        indexed[timestamp] = event
    return [indexed[timestamp] for timestamp in sorted(indexed)]
