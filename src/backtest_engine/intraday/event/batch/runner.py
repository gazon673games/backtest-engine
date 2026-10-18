from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.data.asset_schema import normalize_intraday_asset_frame
from backtest_engine.data.catalog import DataCatalogEntry
from backtest_engine.data.dataset import load_dataset
from backtest_engine.intraday.engine import validate_intraday_dataset
from backtest_engine.intraday.event.batch.cache import (
    deduplicate_parameter_sets,
    reusable_run_manifest,
)
from backtest_engine.intraday.event.batch.case_config import (
    build_case_config,
    resolved_dataset_sources,
)
from backtest_engine.intraday.event.batch.config import (
    IntradayEventBatchConfig,
    IntradayEventBatchResult,
)
from backtest_engine.intraday.event.batch.execution import (
    execution_config,
    generate_intraday_event_run_id,
    resolve_execution_for_dataset,
    run_id_engine_params,
)
from backtest_engine.intraday.event.batch.outputs import (
    batch_result,
    failed_row,
    summary_row,
    write_batch_outputs,
    write_experiment_manifest,
    write_json,
)
from backtest_engine.intraday.event.batch.parallel import run_parallel_cases
from backtest_engine.intraday.event.batch.parsing import (
    expand_intraday_event_grid,
    load_intraday_event_batch_config,
    parse_intraday_event_grid_options,
    safe_path_part,
    split_intraday_event_params,
)
from backtest_engine.intraday.event.data import resolve_event_config
from backtest_engine.intraday.event.engine import run_intraday_event_backtest_frame

__all__ = [
    "IntradayEventBatchConfig",
    "IntradayEventBatchResult",
    "expand_intraday_event_grid",
    "generate_intraday_event_run_id",
    "load_intraday_event_batch_config",
    "parse_intraday_event_grid_options",
    "run_intraday_event_batch",
    "split_intraday_event_params",
]


def run_intraday_event_batch(config: IntradayEventBatchConfig) -> IntradayEventBatchResult:
    experiment_dir = config.output_root / safe_path_part(config.experiment_id)
    runs_dir = experiment_dir / "runs"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    parameter_sets, duplicate_count = deduplicate_parameter_sets(
        expand_intraday_event_grid(config.grid)
    )
    dataset_sources = resolved_dataset_sources(config)
    skipped_duplicate_count = duplicate_count * len(dataset_sources)
    summary_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    write_json(experiment_dir / "experiment_config.json", config.model_dump(mode="json"))

    for dataset_path, catalog_entry in dataset_sources:
        loaded = load_batch_dataset(config, dataset_path, failed_rows)
        if loaded is None:
            if not config.continue_on_error:
                break
            continue
        df, dataset_validation = loaded
        should_continue = run_parameter_grid(
            config,
            runs_dir,
            dataset_path,
            catalog_entry,
            df,
            dataset_validation,
            parameter_sets,
            summary_rows,
            failed_rows,
        )
        if not should_continue:
            break

    write_batch_outputs(experiment_dir, summary_rows, failed_rows)
    write_experiment_manifest(
        experiment_dir,
        config,
        summary_rows,
        failed_rows,
        skipped_duplicate_count=skipped_duplicate_count,
    )
    return batch_result(
        experiment_dir,
        summary_rows,
        failed_rows,
        skipped_duplicate_count=skipped_duplicate_count,
    )


def load_batch_dataset(
    config: IntradayEventBatchConfig,
    dataset_path: Path,
    failed_rows: list[dict[str, Any]],
) -> tuple[pl.DataFrame, dict[str, Any]] | None:
    try:
        df = normalize_intraday_asset_frame(load_dataset(dataset_path))
        return df, validate_intraday_dataset(df, dataset_path)
    except Exception as exc:
        failed_rows.append(failed_row(config, dataset_path, {"dataset": str(dataset_path)}, exc))
        return None


def run_parameter_grid(
    config: IntradayEventBatchConfig,
    runs_dir: Path,
    dataset_path: Path,
    catalog_entry: DataCatalogEntry | None,
    df: pl.DataFrame,
    dataset_validation: dict[str, Any],
    parameter_sets: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> bool:
    def execute_case(
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, str] | None]:
        try:
            return (
                params,
                run_batch_case(
                    config,
                    runs_dir,
                    dataset_path,
                    catalog_entry,
                    df,
                    dataset_validation,
                    params,
                ),
                None,
            )
        except Exception as exc:
            return params, None, failed_row(config, dataset_path, params, exc)

    if config.max_workers > 1 and config.continue_on_error and len(parameter_sets) > 1:
        del df
        results = run_parallel_cases(
            config,
            runs_dir,
            dataset_path,
            catalog_entry,
            parameter_sets,
        )
    else:
        results = (execute_case(params) for params in parameter_sets)
    for _params, row, failure in results:
        if failure is not None:
            failed_rows.append(failure)
            if not config.continue_on_error:
                return False
        elif row is not None:
            summary_rows.append(row)
    return True


def run_batch_case(
    config: IntradayEventBatchConfig,
    runs_dir: Path,
    dataset_path: Path,
    catalog_entry: DataCatalogEntry | None,
    df: pl.DataFrame,
    dataset_validation: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    engine_params, plugin_params = split_intraday_event_params(params)
    execution = resolve_execution_for_dataset(
        execution_config(config, engine_params),
        dataset_validation,
    )
    run_id = generate_intraday_event_run_id(
        dataset_path=dataset_path,
        experiment_id=config.experiment_id,
        dataset_validation=dataset_validation,
        strategy_file=config.strategy_file,
        engine_params=run_id_engine_params(execution),
        plugin_params=plugin_params,
    )
    case_config = build_case_config(
        config,
        runs_dir,
        dataset_path,
        catalog_entry,
        run_id,
        execution,
        plugin_params,
    )
    case_config = resolve_event_config(case_config, dataset_validation)
    from backtest_engine.research.experiment_identity import (
        canonical_fingerprint,
        event_run_fingerprint,
    )

    digest = canonical_fingerprint(
        {"case": run_id, "effective": event_run_fingerprint(case_config, dataset_validation)}
    )[:16]
    run_id = f"{run_id.rsplit('_', 1)[0]}_{digest}"
    case_config = case_config.model_copy(
        update={"output": case_config.output.model_copy(update={"run_id": run_id})}
    )
    cached = (
        reusable_run_manifest(
            case_config,
            dataset_validation,
            verify_output_hashes=config.verify_cached_output_hashes,
        )
        if config.resume_completed
        else None
    )
    if cached is not None:
        return summary_row(
            config=config,
            run_id=run_id,
            run_path=case_config.output.root / case_config.output.run_id,
            dataset_path=dataset_path,
            dataset_validation=dataset_validation,
            quality_approval=dict(cached.get("quality_approval") or {}),
            execution=execution,
            plugin_params=plugin_params,
            metrics=dict(cached.get("metrics") or {}),
            cache_status="hit",
        )
    if (case_config.output.root / case_config.output.run_id).exists():
        from uuid import uuid4

        run_id = f"{run_id}_retry_{uuid4().hex[:8]}"
        case_config = case_config.model_copy(
            update={"output": case_config.output.model_copy(update={"run_id": run_id})}
        )
    result = run_intraday_event_backtest_frame(case_config, df, dataset_validation)
    return summary_row(
        config=config,
        run_id=run_id,
        run_path=result.output_dir,
        dataset_path=dataset_path,
        dataset_validation=dataset_validation,
        quality_approval=result.quality_approval,
        execution=execution,
        plugin_params=plugin_params,
        metrics=result.metrics,
        cache_status="miss",
    )
