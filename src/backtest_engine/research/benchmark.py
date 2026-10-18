from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from backtest_engine.core.exceptions import ReportingError


@dataclass(frozen=True)
class DatasetBenchmarkConfig:
    dataset_path: Path
    output_path: Path
    iterations: int = 3


@dataclass(frozen=True)
class DatasetBenchmarkResult:
    output_path: Path
    row_count: int
    best_seconds: float
    rows_per_second: float


def benchmark_dataset(config: DatasetBenchmarkConfig) -> DatasetBenchmarkResult:
    if config.iterations <= 0:
        raise ReportingError("iterations must be positive")
    if not config.dataset_path.exists():
        raise ReportingError(f"Dataset does not exist: {config.dataset_path}")
    timings = [read_count_seconds(config.dataset_path) for _ in range(config.iterations)]
    row_count = dataset_row_count(config.dataset_path)
    best = min(timings)
    rows_per_second = row_count / best if best > 0.0 else 0.0
    payload = {
        "dataset_path": str(config.dataset_path),
        "iterations": config.iterations,
        "benchmark_kind": "dataset_materialization",
        "row_count": row_count,
        "timings_seconds": timings,
        "best_seconds": best,
        "rows_per_second": rows_per_second,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return DatasetBenchmarkResult(config.output_path, row_count, best, rows_per_second)


def read_count_seconds(path: Path) -> float:
    started = time.perf_counter()
    if path.suffix.lower() == ".parquet":
        pl.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        pl.read_csv(path)
    else:
        raise ReportingError(f"Unsupported benchmark dataset format: {path}")
    return time.perf_counter() - started


def dataset_row_count(path: Path) -> int:
    if path.suffix.lower() == ".parquet":
        return int(pl.scan_parquet(path).select(pl.len()).collect().item())
    return int(pl.scan_csv(path).select(pl.len()).collect().item())
