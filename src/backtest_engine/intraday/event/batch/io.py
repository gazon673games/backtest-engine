"""Columnar and JSON writers for batch-level results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.intraday.event.batch.parsing import optional_string


def write_table(
    base_path: Path,
    rows: list[dict[str, Any]],
    base_schema: dict[str, pl.DataType],
) -> None:
    if rows:
        schema = {
            **{key: dtype for key, dtype in base_schema.items() if all(key in row for row in rows)},
            **{key: pl.String for row in rows for key in row if key not in base_schema},
        }
        normalized_rows = [stringify_dynamic_columns(row, base_schema) for row in rows]
        frame = pl.DataFrame(normalized_rows, schema=schema, orient="row")
    else:
        frame = pl.DataFrame(rows, schema=base_schema, orient="row")
    frame.write_csv(base_path.with_suffix(".csv"))
    frame.write_parquet(base_path.with_suffix(".parquet"))
    write_json(base_path.with_suffix(".json"), rows)


def stringify_dynamic_columns(
    row: dict[str, Any],
    base_schema: dict[str, pl.DataType],
) -> dict[str, Any]:
    return {
        key: (optional_string(value) if key not in base_schema else value)
        for key, value in row.items()
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
