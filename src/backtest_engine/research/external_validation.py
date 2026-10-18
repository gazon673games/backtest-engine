from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import ReportingError


@dataclass(frozen=True)
class ExternalValidationConfig:
    backtest_events_path: Path
    statement_path: Path
    output_path: Path
    tolerance: float = 0.01
    match_columns: tuple[str, ...] = ("ts",)
    row_metric_columns: tuple[str, ...] = ("net_pnl", "cost_cash", "equity_cash")


@dataclass(frozen=True)
class ExternalValidationResult:
    output_path: Path
    status: str
    break_count: int
    total_backtest_pnl: float
    total_statement_pnl: float


def validate_against_statement(config: ExternalValidationConfig) -> ExternalValidationResult:
    if not isfinite(config.tolerance) or config.tolerance < 0.0:
        raise ReportingError("Statement tolerance must be finite and non-negative")
    events = read_table(config.backtest_events_path)
    statement = read_table(config.statement_path)
    required = set(config.match_columns) | set(config.row_metric_columns) | {"net_pnl", "cost_cash"}
    for label, frame in (("backtest", events), ("statement", statement)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ReportingError(f"{label} statement is missing columns: {', '.join(missing)}")
        for column in required:
            if frame[column].null_count():
                raise ReportingError(f"{label}.{column} contains null values")
        for column in required - set(config.match_columns):
            if not all(isfinite(float(value)) for value in frame[column]):
                raise ReportingError(f"{label}.{column} must contain finite numbers")
    event_pnl = total_column(events, "net_pnl")
    statement_pnl = total_column(statement, "net_pnl")
    event_fee = total_column(events, "cost_cash")
    statement_fee = total_column(statement, "cost_cash")
    breaks = []
    add_break(breaks, "net_pnl", event_pnl, statement_pnl, config.tolerance)
    add_break(breaks, "cost_cash", event_fee, statement_fee, config.tolerance)
    if "equity_cash" in events.columns and "equity_cash" in statement.columns:
        add_break(
            breaks,
            "final_equity_cash",
            float(events.get_column("equity_cash")[-1]),
            float(statement.get_column("equity_cash")[-1]),
            config.tolerance,
        )
    breaks.extend(row_level_breaks(events, statement, config))
    payload = {
        "status": "approved" if not breaks else "rejected",
        "tolerance": config.tolerance,
        "break_count": len(breaks),
        "breaks": breaks,
        "total_backtest_pnl": event_pnl,
        "total_statement_pnl": statement_pnl,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ExternalValidationResult(
        output_path=config.output_path,
        status=payload["status"],
        break_count=len(breaks),
        total_backtest_pnl=event_pnl,
        total_statement_pnl=statement_pnl,
    )


def read_table(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise ReportingError(f"Validation table does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    raise ReportingError(f"Validation table must be CSV or Parquet: {path}")


def total_column(frame: pl.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(frame.get_column(column).fill_null(0.0).sum())


def add_break(
    breaks: list[dict[str, Any]],
    metric: str,
    backtest_value: float,
    statement_value: float,
    tolerance: float,
) -> None:
    diff = backtest_value - statement_value
    if not isfinite(diff) or abs(diff) > tolerance:
        breaks.append(
            {
                "metric": metric,
                "backtest": backtest_value,
                "statement": statement_value,
                "diff": diff,
            }
        )


def row_level_breaks(
    events: pl.DataFrame,
    statement: pl.DataFrame,
    config: ExternalValidationConfig,
) -> list[dict[str, Any]]:
    match_columns = [
        column
        for column in config.match_columns
        if column in events.columns and column in statement.columns
    ]
    metric_columns = [
        column
        for column in config.row_metric_columns
        if column in events.columns and column in statement.columns
    ]
    if not match_columns or not metric_columns:
        return []
    left = aggregate_rows(events, match_columns, metric_columns)
    right = aggregate_rows(statement, match_columns, metric_columns)
    breaks = []
    for key in sorted(set(left).union(right), key=str):
        if key not in left or key not in right:
            breaks.append(
                {
                    "metric": "row_presence",
                    "key": list(key),
                    "backtest": key in left,
                    "statement": key in right,
                }
            )
            continue
        for metric in metric_columns:
            add_row_break(
                breaks,
                key,
                metric,
                left[key][metric],
                right[key][metric],
                config.tolerance,
            )
    return breaks


def aggregate_rows(
    frame: pl.DataFrame,
    match_columns: list[str],
    metric_columns: list[str],
) -> dict[tuple[str, ...], dict[str, float]]:
    result: dict[tuple[str, ...], dict[str, float]] = {}
    for row in frame.iter_rows(named=True):
        key = tuple(normalized_key(row.get(column)) for column in match_columns)
        metrics = result.setdefault(key, {column: 0.0 for column in metric_columns})
        for column in metric_columns:
            value = row.get(column)
            if value is not None:
                metrics[column] += float(value)
    return result


def add_row_break(
    breaks: list[dict[str, Any]],
    key: tuple[str, ...],
    metric: str,
    backtest_value: float,
    statement_value: float,
    tolerance: float,
) -> None:
    diff = backtest_value - statement_value
    if not isfinite(diff) or abs(diff) > tolerance:
        breaks.append(
            {
                "metric": f"row.{metric}",
                "key": list(key),
                "backtest": backtest_value,
                "statement": statement_value,
                "diff": diff,
            }
        )


def normalized_key(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)
