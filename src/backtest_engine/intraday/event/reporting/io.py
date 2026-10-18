from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import ReportingError


def read_required_table(experiment_path: Path, name: str) -> pl.DataFrame:
    for suffix in (".parquet", ".csv", ".json"):
        path = experiment_path / f"{name}{suffix}"
        if path.exists():
            return read_table(path)
    msg = f"Missing required experiment table: {experiment_path / name}.[parquet|csv|json]"
    raise ReportingError(msg)


def read_optional_table(experiment_path: Path, name: str) -> pl.DataFrame:
    for suffix in (".parquet", ".csv", ".json"):
        path = experiment_path / f"{name}{suffix}"
        if path.exists():
            return read_table(path)
    return pl.DataFrame()


def read_table(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=1000)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as table_file:
            rows = json.load(table_file)
        return pl.DataFrame(rows, orient="row") if rows else pl.DataFrame()
    msg = f"Unsupported table format: {path}"
    raise ReportingError(msg)


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    return payload if isinstance(payload, dict) else {}


def write_named_outputs(base_path: Path, frame: pl.DataFrame) -> None:
    frame.write_csv(base_path.with_suffix(".csv"))
    frame.write_parquet(base_path.with_suffix(".parquet"))
    write_json(base_path.with_suffix(".json"), [row for row in frame.iter_rows(named=True)])


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
