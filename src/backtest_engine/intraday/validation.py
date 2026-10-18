from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.timeframes import parse_timeframe_minutes
from backtest_engine.data.asset_schema import (
    COMMON_MARKET_COLUMNS,
    compatibility_columns,
    normalize_intraday_asset_frame,
    schema_profile,
    single_text,
    validate_profile_columns,
)
from backtest_engine.intraday.utils import to_iso

REQUIRED_INTRADAY_COLUMNS = COMMON_MARKET_COLUMNS
EXPECTED_INTRADAY_VALUES = {"adjustment_mode": "unadjusted"}


def validate_intraday_dataset(df: pl.DataFrame, dataset_path: Path) -> dict[str, Any]:
    if df.height == 0:
        raise DatasetValidationError("Intraday dataset is empty")

    df = normalize_intraday_asset_frame(df)
    dataset_type = single_text(df, "dataset_type")
    profile = schema_profile(dataset_type)
    validate_profile_columns(df, profile)

    data_kind = optional_single_value(df, "data_kind") or "bars"
    validate_intraday_ts(df, data_kind)
    validate_bar_availability(df)
    validate_intraday_prices(df)
    validate_intraday_activity(df)
    for column in profile.required_columns:
        validate_non_null(df, column)
    timeframe = single_value(df, "timeframe")
    try:
        bar_minutes = parse_timeframe_minutes(timeframe)
    except ValueError as exc:
        raise DatasetValidationError(str(exc)) from exc
    if profile.dataset_type == "nearest_contract":
        validate_single_required_value(df, "adjustment_mode", "unadjusted")

    bar_time_semantics = "open_with_available_at" if "available_at" in df.columns else "legacy_ts"
    return {
        "path": str(dataset_path),
        "validation_status": "accepted",
        "quality_gate_required": False,
        "row_count": df.height,
        "min_ts": to_iso(df.get_column("ts").min()),
        "max_ts": to_iso(df.get_column("ts").max()),
        "continuous_symbol": single_value(df, "continuous_symbol"),
        "timeframe": timeframe,
        "bar_minutes": bar_minutes,
        "dataset_type": dataset_type,
        "adjustment_mode": single_value(df, "adjustment_mode"),
        "asset_class": single_value(df, "asset_class"),
        "instrument_type": single_value(df, "instrument_type"),
        "currency": single_value(df, "currency"),
        "base_currency": optional_single_value(df, "base_currency"),
        "quote_currency": optional_single_value(df, "quote_currency"),
        "settlement_currency": optional_single_value(df, "settlement_currency"),
        "active_contract_count": optional_unique_count(df, "active_contract"),
        "source_contract_symbol_count": optional_unique_count(df, "source_contract_symbol"),
        "compatibility_columns": compatibility_columns(df),
        "allowed_with_explicit_from_date": True,
        "bar_time_semantics": bar_time_semantics,
        "data_kind": data_kind,
    }


def enforce_causal_bar_timestamps(
    validation: dict[str, Any],
    *,
    allow_legacy: bool,
    label: str,
) -> None:
    if validation.get("data_kind") != "bars":
        return
    if validation.get("bar_time_semantics") == "open_with_available_at":
        return
    if not allow_legacy:
        raise DatasetValidationError(
            f"{label} uses legacy ts-only bars. Canonical bar_open_ts, bar_close_ts, "
            "available_at, and is_final columns are required to prevent OHLC lookahead; "
            "set allow_legacy_bar_timestamps=true only for an explicit diagnostic run."
        )


def optional_unique_count(df: pl.DataFrame, column: str) -> int:
    return df.get_column(column).n_unique() if column in df.columns else 0


def optional_single_value(df: pl.DataFrame, column: str) -> str | None:
    if column not in df.columns:
        return None
    values = df.get_column(column).drop_nulls().unique().to_list()
    if not values:
        return None
    if len(values) != 1:
        raise DatasetValidationError(f"{column} must contain at most one non-null value")
    return str(values[0])


def validate_intraday_ts(df: pl.DataFrame, data_kind: str = "bars") -> None:
    ts = df.get_column("ts")
    if ts.null_count() > 0:
        raise DatasetValidationError("ts must be non-null")
    if data_kind == "bars" and ts.n_unique() != df.height:
        raise DatasetValidationError("ts must not contain duplicates")
    if data_kind in {"trades", "quotes", "book"}:
        if "event_id" not in df.columns:
            raise DatasetValidationError(f"{data_kind} dataset requires event_id")
        event_ids = df.get_column("event_id")
        if event_ids.null_count() or event_ids.n_unique() != df.height:
            raise DatasetValidationError("event_id must be non-null and unique")
    if not ts.is_sorted():
        raise DatasetValidationError("ts must be sorted ascending")


def validate_intraday_prices(df: pl.DataFrame) -> None:
    for column in ("open", "high", "low", "close"):
        validate_finite_numeric(df, column, positive=True)
    invalid_ohlc = df.filter(
        (pl.col("high") < pl.max_horizontal("open", "close", "low"))
        | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
    )
    if invalid_ohlc.height:
        raise DatasetValidationError(
            f"OHLC price bounds are inconsistent in {invalid_ohlc.height} rows"
        )


def validate_intraday_activity(df: pl.DataFrame) -> None:
    for column in ("volume", "value"):
        validate_finite_numeric(df, column, positive=False)


def validate_finite_numeric(df: pl.DataFrame, column: str, *, positive: bool) -> None:
    dtype = df.schema[column]
    if not dtype.is_numeric():
        raise DatasetValidationError(f"{column} must be numeric, got {dtype}")
    values = df.get_column(column)
    if values.null_count() > 0:
        raise DatasetValidationError(f"{column} must be non-null")
    invalid = ~pl.col(column).is_finite()
    invalid |= pl.col(column) <= 0.0 if positive else pl.col(column) < 0.0
    invalid_count = df.select(invalid.sum()).item()
    if invalid_count:
        requirement = "finite and positive" if positive else "finite and non-negative"
        raise DatasetValidationError(
            f"{column} must be {requirement}; invalid rows={invalid_count}"
        )


def validate_bar_availability(df: pl.DataFrame) -> None:
    optional = {"bar_open_ts", "bar_close_ts", "available_at", "is_final"}
    present = optional.intersection(df.columns)
    if present and present != optional:
        missing = ", ".join(sorted(optional.difference(present)))
        raise DatasetValidationError(f"bar availability columns are incomplete: {missing}")
    if not present:
        return
    for column in ("bar_open_ts", "bar_close_ts", "available_at", "is_final"):
        validate_non_null(df, column)
    invalid_order = df.filter(
        (pl.col("bar_open_ts") > pl.col("bar_close_ts"))
        | (pl.col("bar_close_ts") > pl.col("available_at"))
    )
    if invalid_order.height:
        raise DatasetValidationError("bar timestamps must satisfy open <= close <= available_at")
    if not bool(df.get_column("is_final").all()):
        raise DatasetValidationError("intraday dataset contains non-final bars")


def validate_non_null(df: pl.DataFrame, column: str) -> None:
    if df.get_column(column).null_count() > 0:
        msg = f"{column} must be non-null"
        raise DatasetValidationError(msg)


def validate_single_required_value(df: pl.DataFrame, column: str, expected: str) -> None:
    column_values = df.get_column(column)
    if column_values.null_count() > 0:
        msg = f"{column} must be non-null and equal {expected}"
        raise DatasetValidationError(msg)
    values = sorted(str(value) for value in column_values.unique().to_list())
    if values != [expected]:
        msg = f"{column} must be {expected}, got {values}"
        raise DatasetValidationError(msg)


def single_value(df: pl.DataFrame, column: str) -> str:
    column_values = df.get_column(column)
    if column_values.null_count() > 0:
        msg = f"{column} must be non-null"
        raise DatasetValidationError(msg)
    values = column_values.unique().to_list()
    if len(values) != 1:
        msg = f"{column} must contain exactly one non-null value"
        raise DatasetValidationError(msg)
    return str(values[0])
