from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.reporting.tables import markdown_table as markdown_table


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def json_text(value: Any) -> str | None:
    if value in (None, {}, []):
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def normalize_path_key(path: Path) -> str:
    try:
        return str(path.resolve()).casefold()
    except OSError:
        return str(path).casefold()


def decision_time_text(execution: dict[str, Any]) -> str | None:
    if "decision_time" in execution:
        return optional_string(execution.get("decision_time"))
    values = execution.get("decision_times")
    if values is None:
        return None
    if isinstance(values, list):
        return ",".join(str(value) for value in values)
    return str(values)


def unique_strings(frame: pl.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(str(value) for value in frame.get_column(column).drop_nulls().unique())


def status_count(frame: pl.DataFrame, status: str) -> int:
    if frame.height == 0 or "status" not in frame.columns:
        return 0
    return frame.filter(pl.col("status") == status).height


def select_existing_columns(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    existing = [column for column in columns if column in frame.columns]
    return frame.select(existing) if existing else pl.DataFrame()
