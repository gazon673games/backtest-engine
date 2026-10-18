from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from backtest_engine.core.exceptions import ReportingError


def read_table(base_path: Path) -> pl.DataFrame:
    for suffix in (".parquet", ".csv", ".json"):
        path = base_path.with_suffix(suffix)
        if not path.exists():
            continue
        if suffix == ".parquet":
            return pl.read_parquet(path)
        if suffix == ".csv":
            return pl.read_csv(path, infer_schema_length=1000)
        rows = json.loads(path.read_text(encoding="utf-8"))
        return pl.DataFrame(rows, orient="row") if rows else pl.DataFrame()
    raise ReportingError(f"Missing table: {base_path}.[parquet|csv|json]")


def write_table(base_path: Path, frame: pl.DataFrame) -> None:
    frame.write_csv(base_path.with_suffix(".csv"))
    frame.write_parquet(base_path.with_suffix(".parquet"))
    rows = frame.to_dicts() if frame.height else []
    base_path.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
