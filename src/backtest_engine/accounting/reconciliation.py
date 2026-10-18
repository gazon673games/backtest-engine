"""Golden-statement reconciliation for generated portfolio ledgers."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError

DEFAULT_KEYS = ("ts", "symbol", "currency")
DEFAULT_VALUES = ("net_cash_flow", "cash_balance", "equity_cash")


def reconcile_ledger(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    *,
    tolerance: float = 0.01,
    key_columns: tuple[str, ...] = DEFAULT_KEYS,
    value_columns: tuple[str, ...] = DEFAULT_VALUES,
) -> dict[str, Any]:
    if not isfinite(tolerance) or tolerance < 0.0:
        raise DatasetValidationError("Reconciliation tolerance must be finite and non-negative")
    if not key_columns or not value_columns:
        raise DatasetValidationError("Reconciliation requires key and monetary columns")
    available_values = value_columns
    required = set(key_columns) | set(value_columns)
    for label, frame in (("actual", actual), ("expected", expected)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DatasetValidationError(
                f"Reconciliation {label} ledger is missing columns: {', '.join(missing)}"
            )
        for column in required:
            if frame[column].null_count():
                raise DatasetValidationError(
                    f"Reconciliation {label}.{column} contains null values"
                )
        for column in value_columns:
            try:
                valid = all(isfinite(float(value)) for value in frame[column])
            except (ValueError, TypeError):
                valid = False
            if not valid:
                raise DatasetValidationError(
                    f"Reconciliation {label}.{column} must contain finite numbers"
                )
        duplicate_count = frame.group_by(list(key_columns)).len().filter(pl.col("len") > 1).height
        if duplicate_count:
            raise DatasetValidationError(
                f"Reconciliation {label} ledger has {duplicate_count} duplicate keys"
            )
    actual_rows = keyed_rows(actual, key_columns)
    expected_rows = keyed_rows(expected, key_columns)
    breaks: list[dict[str, Any]] = []
    for key in sorted(set(actual_rows) | set(expected_rows), key=str):
        if key not in actual_rows:
            breaks.append(break_row(key_columns, key, "missing_actual"))
            continue
        if key not in expected_rows:
            breaks.append(break_row(key_columns, key, "unexpected_actual"))
            continue
        for column in available_values:
            actual_value = optional_float(actual_rows[key].get(column))
            expected_value = optional_float(expected_rows[key].get(column))
            diff = actual_value - expected_value
            if abs(diff) > tolerance:
                breaks.append(
                    break_row(
                        key_columns,
                        key,
                        "value_mismatch",
                        field=column,
                        expected=expected_value,
                        actual=actual_value,
                        diff=diff,
                    )
                )
    return {
        "status": "reconciled" if not breaks else "failed",
        "tolerance": tolerance,
        "actual_row_count": actual.height,
        "expected_row_count": expected.height,
        "compared_value_columns": list(available_values),
        "break_count": len(breaks),
        "breaks": breaks,
    }


def reconcile_ledger_path(
    actual: pl.DataFrame,
    expected_path: Path,
    *,
    tolerance: float = 0.01,
    fail_on_break: bool = True,
) -> dict[str, Any]:
    if not expected_path.is_file():
        raise DatasetValidationError(f"Reconciliation statement does not exist: {expected_path}")
    expected = read_statement(expected_path)
    result = reconcile_ledger(actual, expected, tolerance=tolerance)
    result["expected_path"] = str(expected_path)
    if fail_on_break:
        assert_reconciled(result)
    return result


def assert_reconciled(result: dict[str, Any]) -> None:
    if result["status"] != "reconciled":
        raise DatasetValidationError(
            f"Portfolio ledger reconciliation failed with {result['break_count']} breaks"
        )


def read_statement(path: Path) -> pl.DataFrame:
    try:
        if path.suffix.lower() == ".csv":
            return pl.read_csv(path, try_parse_dates=True)
        if path.suffix.lower() == ".parquet":
            return pl.read_parquet(path)
    except Exception as exc:
        raise DatasetValidationError(f"Failed to read reconciliation statement: {path}") from exc
    raise DatasetValidationError("Reconciliation statement must be CSV or Parquet")


def keyed_rows(
    frame: pl.DataFrame, key_columns: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, Any]]:
    return {
        tuple(normalized_key(row[column]) for column in key_columns): row
        for row in frame.iter_rows(named=True)
    }


def normalized_key(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def break_row(
    columns: tuple[str, ...],
    key: tuple[str, ...],
    reason: str,
    **values: Any,
) -> dict[str, Any]:
    return {"key": dict(zip(columns, key, strict=True)), "reason": reason, **values}


def optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
