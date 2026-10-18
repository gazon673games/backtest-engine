from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.models import DatasetMetadata

REQUIRED_COLUMNS = frozenset(
    {
        "continuous_symbol",
        "timeframe",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
        "active_contract",
        "source_contract_symbol",
        "roll_id",
        "roll_flag",
        "roll_date",
        "adjustment_mode",
        "adjustment_factor",
    }
)


def load_dataset(dataset_path: Path) -> pl.DataFrame:
    if not dataset_path.exists():
        msg = f"Dataset file does not exist: {dataset_path}"
        raise DatasetValidationError(msg)
    if not dataset_path.is_file():
        msg = f"Dataset path is not a file: {dataset_path}"
        raise DatasetValidationError(msg)
    try:
        return pl.read_parquet(dataset_path)
    except Exception as exc:  # pragma: no cover - polars supplies the exact detail
        msg = f"Failed to read Parquet dataset: {dataset_path}"
        raise DatasetValidationError(msg) from exc


def validate_dataset(df: pl.DataFrame, dataset_path: Path) -> DatasetMetadata:
    if df.height == 0:
        raise DatasetValidationError("Dataset is empty")

    missing_columns = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing_columns:
        msg = f"Dataset is missing required columns: {', '.join(missing_columns)}"
        raise DatasetValidationError(msg)

    _validate_ts(df)
    _validate_close(df)
    _validate_timeframe(df)

    continuous_symbol = _single_value(df, "continuous_symbol")
    adjustment_mode = _single_value(df, "adjustment_mode")

    return DatasetMetadata(
        path=str(dataset_path),
        continuous_symbol=continuous_symbol,
        adjustment_mode=adjustment_mode,
        row_count=df.height,
        min_ts=_to_iso(df.get_column("ts").min()),
        max_ts=_to_iso(df.get_column("ts").max()),
    )


def _validate_ts(df: pl.DataFrame) -> None:
    ts = df.get_column("ts")
    if ts.null_count() > 0:
        raise DatasetValidationError("ts must be non-null")
    if ts.n_unique() != df.height:
        raise DatasetValidationError("ts must not contain duplicates")
    if not ts.is_sorted():
        raise DatasetValidationError("ts must be sorted ascending")


def _validate_close(df: pl.DataFrame) -> None:
    close = df.get_column("close")
    if close.null_count() > 0:
        raise DatasetValidationError("close must be non-null")
    close_dtype = df.schema["close"]
    if not close_dtype.is_numeric():
        msg = f"close must be numeric, got {close_dtype}"
        raise DatasetValidationError(msg)
    invalid_count = df.select(
        (~pl.col("close").is_finite() | (pl.col("close") <= 0.0)).sum()
    ).item()
    if invalid_count:
        raise DatasetValidationError(
            f"close must be finite and positive; invalid rows={invalid_count}"
        )


def _validate_timeframe(df: pl.DataFrame) -> None:
    timeframe = df.get_column("timeframe")
    if timeframe.null_count() > 0:
        raise DatasetValidationError("timeframe must be non-null")
    values = sorted(str(value) for value in timeframe.unique().to_list())
    if values != ["1d"]:
        msg = f"timeframe must be 1d, got {values}"
        raise DatasetValidationError(msg)


def _single_value(df: pl.DataFrame, column: str) -> str:
    values = df.get_column(column).drop_nulls().unique().to_list()
    if len(values) != 1:
        msg = f"{column} must contain exactly one non-null value"
        raise DatasetValidationError(msg)
    return str(values[0])


def _to_iso(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
