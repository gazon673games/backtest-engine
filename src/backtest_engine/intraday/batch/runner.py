from __future__ import annotations

from typing import Any

import polars as pl

from backtest_engine.data.asset_schema import normalize_intraday_asset_frame
from backtest_engine.data.dataset import load_dataset
from backtest_engine.intraday.batch.config import (
    IntradayBatchConfig,
    IntradayBatchResult,
    IntradayBatchRunResult,
)
from backtest_engine.intraday.batch.outputs import (
    failed_row,
    summary_row,
    trade_summary_metrics,
    write_batch_outputs,
    write_json,
)
from backtest_engine.intraday.batch.parsing import (
    expand_intraday_grid,
    generate_intraday_run_id,
    load_intraday_batch_config,
    optional_int,
    parse_date,
    parse_intraday_grid_options,
    parse_time,
    required_float,
    required_int,
    safe_path_part,
    split_intraday_params,
)
from backtest_engine.intraday.engine import (
    IntradayBacktestConfig,
    IntradayDatasetConfig,
    IntradayExecutionConfig,
    IntradayOutputConfig,
    IntradayStrategyConfig,
    run_intraday_backtest_frame,
    validate_intraday_dataset,
)

__all__ = [
    "IntradayBatchConfig",
    "IntradayBatchResult",
    "IntradayBatchRunResult",
    "expand_intraday_grid",
    "generate_intraday_run_id",
    "load_intraday_batch_config",
    "parse_intraday_grid_options",
    "run_intraday_batch",
    "split_intraday_params",
]


def run_intraday_batch(config: IntradayBatchConfig) -> IntradayBatchResult:
    experiment_dir = config.output_root / safe_path_part(config.experiment_id)
    runs_dir = experiment_dir / "runs"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_intraday_asset_frame(load_dataset(config.dataset_path))
    dataset_validation = validate_intraday_dataset(df, config.dataset_path)
    summary_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    write_json(experiment_dir / "experiment_config.json", config.model_dump(mode="json"))

    for params in expand_intraday_grid(config.grid):
        if not run_intraday_batch_case(
            config,
            runs_dir,
            df,
            dataset_validation,
            params,
            summary_rows,
            failed_rows,
        ):
            break

    write_batch_outputs(experiment_dir, summary_rows, failed_rows)
    return IntradayBatchResult(
        output_dir=experiment_dir,
        successful_run_count=len(summary_rows),
        failed_run_count=len(failed_rows),
        summary_rows=summary_rows,
        failed_rows=failed_rows,
    )


def run_intraday_batch_case(
    config: IntradayBatchConfig,
    runs_dir: Any,
    df: pl.DataFrame,
    dataset_validation: dict[str, Any],
    params: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> bool:
    try:
        engine_params, plugin_params = split_intraday_params(params)
        execution = resolve_execution_for_dataset(
            execution_config(config, engine_params),
            dataset_validation,
        )
        run_id = generate_intraday_run_id(
            experiment_id=config.experiment_id,
            dataset_validation=dataset_validation,
            strategy_file=config.strategy_file,
            engine_params=run_id_engine_params(execution),
            plugin_params=plugin_params,
        )
        result = run_intraday_backtest_frame(
            run_config(config, runs_dir, run_id, execution, plugin_params),
            df,
            dataset_validation,
        )
        trades = pl.read_parquet(result.output_dir / "trades.parquet")
        summary_rows.append(
            summary_row(
                config=config,
                run_id=run_id,
                run_path=result.output_dir,
                dataset_validation=dataset_validation,
                execution=execution,
                plugin_params=plugin_params,
                metrics=result.metrics,
                trade_metrics=trade_summary_metrics(trades),
            )
        )
    except Exception as exc:
        failed_rows.append(failed_row(config, params, exc))
        return config.continue_on_error
    return True


def run_config(
    config: IntradayBatchConfig,
    runs_dir: Any,
    run_id: str,
    execution: IntradayExecutionConfig,
    plugin_params: dict[str, Any],
) -> IntradayBacktestConfig:
    return IntradayBacktestConfig(
        dataset=IntradayDatasetConfig(
            path=config.dataset_path,
            quality_approval_path=config.quality_approval_path,
            quality_approval_required=config.quality_approval_required,
            min_usable_decision_day_pct=config.min_usable_decision_day_pct,
            allow_legacy_bar_timestamps=config.allow_legacy_bar_timestamps,
        ),
        strategy=IntradayStrategyConfig(
            strategy_file=config.strategy_file,
            strategy_params={key: str(value) for key, value in plugin_params.items()},
        ),
        execution=execution,
        output=IntradayOutputConfig(root=runs_dir, run_id=run_id),
    )


def execution_config(
    config: IntradayBatchConfig,
    engine_params: dict[str, Any],
) -> IntradayExecutionConfig:
    return IntradayExecutionConfig(
        from_date=parse_date(engine_params.get("from_date", config.from_date)),
        decision_time=parse_time(engine_params.get("decision_time", config.decision_time)),
        decision_tolerance_minutes=optional_int(
            engine_params.get(
                "decision_tolerance_minutes",
                config.decision_tolerance_minutes,
            )
        ),
        minimum_bars_since_session_start=required_int(
            engine_params.get(
                "minimum_bars_since_session_start",
                config.minimum_bars_since_session_start,
            ),
            "minimum_bars_since_session_start",
        ),
        commission_bps=required_float(
            engine_params.get("commission_bps", config.commission_bps),
            "commission_bps",
        ),
        slippage_bps=required_float(
            engine_params.get("slippage_bps", config.slippage_bps),
            "slippage_bps",
        ),
    )


def resolve_execution_for_dataset(
    execution: IntradayExecutionConfig,
    dataset_validation: dict[str, Any],
) -> IntradayExecutionConfig:
    if execution.decision_tolerance_minutes is not None:
        return execution
    return execution.model_copy(
        update={"decision_tolerance_minutes": int(dataset_validation["bar_minutes"])}
    )


def run_id_engine_params(execution: IntradayExecutionConfig) -> dict[str, Any]:
    return {
        "from_date": execution.from_date,
        "decision_time": execution.decision_time,
        "decision_tolerance_minutes": execution.decision_tolerance_minutes,
        "minimum_bars_since_session_start": execution.minimum_bars_since_session_start,
        "commission_bps": execution.commission_bps,
        "slippage_bps": execution.slippage_bps,
    }
