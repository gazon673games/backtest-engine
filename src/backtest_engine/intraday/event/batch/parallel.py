from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.data.asset_schema import normalize_intraday_asset_frame
from backtest_engine.data.catalog import DataCatalogEntry
from backtest_engine.data.dataset import load_dataset
from backtest_engine.intraday.engine import validate_intraday_dataset
from backtest_engine.intraday.event.batch.config import IntradayEventBatchConfig
from backtest_engine.intraday.event.batch.outputs import failed_row

CaseResult = tuple[dict[str, Any], dict[str, Any] | None, dict[str, str] | None]

_WORKER_FRAME: pl.DataFrame | None = None
_WORKER_VALIDATION: dict[str, Any] | None = None


def run_parallel_cases(
    config: IntradayEventBatchConfig,
    runs_dir: Path,
    dataset_path: Path,
    catalog_entry: DataCatalogEntry | None,
    parameter_sets: list[dict[str, Any]],
) -> list[CaseResult]:
    worker_count = min(config.max_workers, len(parameter_sets))
    tasks = [(config, runs_dir, dataset_path, catalog_entry, params) for params in parameter_sets]
    with ProcessPoolExecutor(
        mp_context=get_context("spawn"),
        max_workers=worker_count,
        initializer=_initialize_worker,
        initargs=(dataset_path,),
    ) as executor:
        return list(executor.map(_execute_case, tasks))


def _initialize_worker(dataset_path: Path) -> None:
    global _WORKER_FRAME, _WORKER_VALIDATION
    _WORKER_FRAME = normalize_intraday_asset_frame(load_dataset(dataset_path))
    _WORKER_VALIDATION = validate_intraday_dataset(_WORKER_FRAME, dataset_path)


def _execute_case(
    task: tuple[
        IntradayEventBatchConfig,
        Path,
        Path,
        DataCatalogEntry | None,
        dict[str, Any],
    ],
) -> CaseResult:
    config, runs_dir, dataset_path, catalog_entry, params = task
    if _WORKER_FRAME is None or _WORKER_VALIDATION is None:
        raise RuntimeError("parameter-grid worker dataset was not initialized")
    try:
        from backtest_engine.intraday.event.batch.runner import run_batch_case

        row = run_batch_case(
            config,
            runs_dir,
            dataset_path,
            catalog_entry,
            _WORKER_FRAME,
            _WORKER_VALIDATION,
            params,
        )
        return params, row, None
    except Exception as exc:
        return params, None, failed_row(config, dataset_path, params, exc)
