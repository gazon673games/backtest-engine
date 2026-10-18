"""Resolve Workbench account defaults from the explicit dataset identity."""

from __future__ import annotations

from typing import Any

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.market_identity import account_currency, annualization_days
from backtest_engine.workbench.models import WorkbenchRunRequest


def account_defaults(request: WorkbenchRunRequest) -> dict[str, Any]:
    schema = pl.read_parquet_schema(request.dataset_path)
    columns = [
        name
        for name in (
            "asset_class",
            "instrument_type",
            "settlement_currency",
            "quote_currency",
            "currency",
        )
        if name in schema
    ]
    expressions = [
        item
        for name in columns
        for item in (
            pl.col(name).drop_nulls().first().alias(name),
            pl.col(name).drop_nulls().n_unique().alias(f"{name}_count"),
        )
    ]
    identity = pl.scan_parquet(request.dataset_path).select(expressions).collect()
    values = identity.row(0, named=True) if identity.height else {}
    if any(values.get(f"{name}_count", 0) > 1 for name in columns):
        raise DatasetValidationError(
            "Workbench requires a dataset with one market/currency identity"
        )
    perpetual = values.get("instrument_type") == "perpetual"
    currency = (
        values.get("settlement_currency") or values.get("quote_currency") or values.get("currency")
    )
    return {
        "base_currency": account_currency(request.base_currency, [currency]),
        "allow_fractional_contracts": (
            True
            if request.allow_fractional_contracts is None
            else request.allow_fractional_contracts
        ),
        "trading_days_per_year": annualization_days(
            request.trading_days_per_year, [request.calendar_path]
        ),
        "require_mark_price": perpetual and request.quality_approval_required,
        "require_margin_tiers": perpetual and request.quality_approval_required,
    }
