"""Exchange-neutral resolution of explicit instrument and reporting inputs."""

import json
import tomllib
from pathlib import Path
from typing import Any

from backtest_engine.core.exceptions import DatasetValidationError


def identity_updates(instrument: Any, validation: dict[str, Any]) -> dict[str, Any]:
    updates = {}
    for field in (
        "asset_class",
        "instrument_type",
        "base_currency",
        "quote_currency",
        "settlement_currency",
    ):
        if field not in instrument.model_fields_set and validation.get(field) is not None:
            updates[field] = validation[field]
    quote = validation.get("quote_currency") or validation.get("currency")
    settlement = validation.get("settlement_currency") or quote
    for field, value in (("price_currency", quote), ("pnl_currency", settlement)):
        if getattr(instrument, field) is None and value:
            updates[field] = value
    if "contract_type" not in instrument.model_fields_set:
        kind = validation.get("instrument_type") or instrument.instrument_type
        updates["contract_type"] = "spot" if kind == "spot" else "linear"
    return updates


def account_currency(explicit: str | None, currencies: list[str | None]) -> str:
    if explicit:
        return explicit
    unique = {value for value in currencies if value}
    if len(unique) != 1 or any(value is None for value in currencies):
        raise DatasetValidationError(
            f"portfolio.base_currency is required: settlement currencies={sorted(unique)}"
        )
    return unique.pop()


def annualization_days(explicit: int | None, calendars: list[Path | None]) -> int | None:
    if explicit is not None:
        return explicit
    values = []
    for path in calendars:
        if path is None:
            return None
        with path.open("rb") as stream:
            calendar = tomllib.load(stream) if path.suffix.lower() == ".toml" else json.load(stream)
        value = calendar.get("trading_days_per_year")
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DatasetValidationError(
                f"{path}: trading_days_per_year must be a positive integer"
            )
        values.append(value)
    unique = set(values)
    if len(unique) > 1:
        raise DatasetValidationError(
            "execution.trading_days_per_year is required for calendars with different annualization"
        )
    return next(iter(unique), None)
