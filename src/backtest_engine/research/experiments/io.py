from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import ReportingError
from backtest_engine.research.experiments.config import REGISTRY_COLUMNS, OutputFormat


def load_experiment_registry(registry_path: Path) -> pl.DataFrame:
    if registry_path.suffix == ".parquet":
        return pl.read_parquet(registry_path)
    if registry_path.suffix == ".csv":
        return pl.read_csv(registry_path, infer_schema_length=1000)
    if registry_path.suffix == ".json":
        with registry_path.open("r", encoding="utf-8") as registry_file:
            rows = json.load(registry_file)
        return registry_frame(rows if isinstance(rows, list) else [])
    msg = f"Unsupported experiment registry format: {registry_path}"
    raise ReportingError(msg)


def registry_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame([], schema={column: pl.String for column in REGISTRY_COLUMNS})
    return pl.DataFrame(rows, orient="row").select(REGISTRY_COLUMNS)


def output_formats(output_format: OutputFormat) -> set[str]:
    if output_format == "all":
        return {"md", "json", "csv", "parquet"}
    return {output_format}


def write_named_outputs(
    base_path: Path,
    frame: pl.DataFrame,
    formats: set[str],
) -> Path | None:
    first_path = None
    table_formats = formats.intersection({"csv", "parquet", "json"})
    if "parquet" in table_formats:
        first_path = first_path or base_path.with_suffix(".parquet")
        frame.write_parquet(base_path.with_suffix(".parquet"))
    if "csv" in table_formats:
        first_path = first_path or base_path.with_suffix(".csv")
        frame.write_csv(base_path.with_suffix(".csv"))
    if "json" in table_formats:
        first_path = first_path or base_path.with_suffix(".json")
        write_json(base_path.with_suffix(".json"), frame.to_dicts())
    return first_path


def output_file_hashes(output_path: Path, prefix: str) -> dict[str, dict[str, str]]:
    paths = [
        path for path in output_path.iterdir() if path.is_file() and path.stem.startswith(prefix)
    ]
    return {path.name: {"path": str(path), "sha256": file_sha256(path)} for path in sorted(paths)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
