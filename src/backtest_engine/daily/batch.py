from __future__ import annotations

import itertools
import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backtest_engine.core.exceptions import QualityGateError
from backtest_engine.core.models import (
    BacktestConfig,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
    StrategyConfig,
)
from backtest_engine.daily.batch_identity import generate_run_id as generate_run_id
from backtest_engine.daily.batch_io import safe_path_part, write_batch_outputs, write_json
from backtest_engine.daily.engine import inspect_dataset, run_backtest
from backtest_engine.strategies.plugins.loader import strategy_file_sha256

FAST_LT_SLOW_CONSTRAINT = "fast_window_lt_slow_window"


class BatchConfig(BaseModel):
    experiment_id: str = Field(min_length=1)
    output_root: Path = Path("data/backtests")
    dataset_paths: list[Path]
    quality_manifest: Path
    strategy_file: Path
    grid: dict[str, list[int]]
    constraints: list[str] = Field(default_factory=list)
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    trading_days_per_year: int | None = Field(default=None, gt=0)
    continue_on_error: bool = True


class BatchResult(BaseModel):
    output_dir: Path
    successful_run_count: int
    failed_run_count: int
    summary_rows: list[dict[str, Any]]
    failed_rows: list[dict[str, Any]]


def run_batch(config: BatchConfig) -> BatchResult:
    experiment_dir = config.output_root / safe_path_part(config.experiment_id)
    runs_dir = experiment_dir / "runs"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    parameter_sets = expand_grid(config.grid, config.constraints)

    write_json(experiment_dir / "experiment_config.json", config.model_dump(mode="json"))

    for dataset_path in config.dataset_paths:
        for params in parameter_sets:
            try:
                inspection = inspect_dataset(dataset_path, config.quality_manifest)
                if not inspection.quality.approved_for_backtest:
                    msg = (
                        "Dataset is not approved for backtesting: "
                        f"status={inspection.quality.status}"
                    )
                    raise QualityGateError(msg)

                run_name = generate_run_id(
                    continuous_symbol=inspection.metadata.continuous_symbol,
                    adjustment_mode=inspection.metadata.adjustment_mode,
                    strategy_file=config.strategy_file,
                    params=params,
                    dataset_path=dataset_path,
                    execution_params={
                        "commission_bps": config.commission_bps,
                        "slippage_bps": config.slippage_bps,
                        "trading_days_per_year": config.trading_days_per_year,
                    },
                )
                backtest_config = BacktestConfig(
                    dataset=DatasetConfig(
                        path=dataset_path,
                        quality_manifest=config.quality_manifest,
                    ),
                    strategy=StrategyConfig(
                        name="plugin",
                        strategy_file=config.strategy_file,
                        strategy_params={key: str(value) for key, value in params.items()},
                    ),
                    execution=ExecutionConfig(
                        commission_bps=config.commission_bps,
                        slippage_bps=config.slippage_bps,
                        trading_days_per_year=config.trading_days_per_year,
                    ),
                    output=OutputConfig(root=runs_dir, run_id=run_name),
                )
                result = run_backtest(backtest_config)
                summary_rows.append(
                    _summary_row(
                        config=config,
                        run_id=run_name,
                        run_path=result.output_dir,
                        dataset_path=dataset_path,
                        continuous_symbol=inspection.metadata.continuous_symbol,
                        adjustment_mode=inspection.metadata.adjustment_mode,
                        params=params,
                        metrics=result.metrics.model_dump(mode="json"),
                    )
                )
            except Exception as exc:
                failed_rows.append(_failed_row(config, dataset_path, params, exc))
                if not config.continue_on_error:
                    write_batch_outputs(experiment_dir, summary_rows, failed_rows)
                    return BatchResult(
                        output_dir=experiment_dir,
                        successful_run_count=len(summary_rows),
                        failed_run_count=len(failed_rows),
                        summary_rows=summary_rows,
                        failed_rows=failed_rows,
                    )

    write_batch_outputs(experiment_dir, summary_rows, failed_rows)
    return BatchResult(
        output_dir=experiment_dir,
        successful_run_count=len(summary_rows),
        failed_run_count=len(failed_rows),
        summary_rows=summary_rows,
        failed_rows=failed_rows,
    )


def load_batch_config(config_path: Path) -> BatchConfig:
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    constraints = [
        key for key, enabled in raw_config.get("constraints", {}).items() if bool(enabled)
    ]
    return BatchConfig(
        experiment_id=raw_config["experiment_id"],
        output_root=raw_config.get("output_root", "data/backtests"),
        dataset_paths=raw_config["dataset_paths"],
        quality_manifest=raw_config["quality_manifest"],
        strategy_file=raw_config["strategy_file"],
        grid={
            key: [int(value) for value in values]
            for key, values in raw_config.get("grid", {}).items()
        },
        constraints=constraints,
        commission_bps=float(raw_config.get("commission_bps", 0.0)),
        slippage_bps=float(raw_config.get("slippage_bps", 0.0)),
        trading_days_per_year=raw_config.get("trading_days_per_year"),
        continue_on_error=bool(raw_config.get("continue_on_error", True)),
    )


def expand_grid(
    grid: dict[str, list[int]],
    constraints: list[str] | None = None,
) -> list[dict[str, int]]:
    if not grid:
        msg = "Batch grid must contain at least one parameter"
        raise ValueError(msg)

    constraints = constraints or []
    keys = sorted(grid)
    combinations = [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[key] for key in keys))
    ]
    return [params for params in combinations if _params_allowed(params, constraints)]


def parse_grid_options(grid_options: list[str]) -> dict[str, list[int]]:
    grid: dict[str, list[int]] = {}
    for raw_grid in grid_options:
        if "=" not in raw_grid:
            msg = f"Invalid --grid value, expected key=v1,v2: {raw_grid}"
            raise ValueError(msg)
        key, raw_values = raw_grid.split("=", 1)
        key = key.strip()
        if not key:
            msg = f"Invalid --grid value, empty key: {raw_grid}"
            raise ValueError(msg)
        if key in grid:
            msg = f"Duplicate --grid key: {key}"
            raise ValueError(msg)
        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not values:
            msg = f"Invalid --grid value, no values for key: {key}"
            raise ValueError(msg)
        try:
            grid[key] = [int(value) for value in values]
        except ValueError as exc:
            msg = f"Grid values must be integers: {raw_grid}"
            raise ValueError(msg) from exc
    return grid


def _params_allowed(params: dict[str, int], constraints: list[str]) -> bool:
    for constraint in constraints:
        if constraint == FAST_LT_SLOW_CONSTRAINT:
            if params.get("fast_window", 0) >= params.get("slow_window", 0):
                return False
        else:
            msg = f"Unsupported batch constraint: {constraint}"
            raise ValueError(msg)
    return True


def _summary_row(
    *,
    config: BatchConfig,
    run_id: str,
    run_path: Path,
    dataset_path: Path,
    continuous_symbol: str,
    adjustment_mode: str,
    params: dict[str, int],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "run_id": run_id,
        "run_path": str(run_path),
        "dataset_path": str(dataset_path),
        "continuous_symbol": continuous_symbol,
        "adjustment_mode": adjustment_mode,
        "strategy_type": "plugin",
        "strategy_file": str(config.strategy_file),
        "strategy_file_sha256": strategy_file_sha256(config.strategy_file),
        "fast_window": params.get("fast_window"),
        "slow_window": params.get("slow_window"),
        "commission_bps": config.commission_bps,
        "slippage_bps": config.slippage_bps,
        "start_ts": metrics["start_ts"],
        "end_ts": metrics["end_ts"],
        "bar_count": metrics["bar_count"],
        "total_return": metrics["total_return"],
        "annualized_return": metrics["annualized_return"],
        "annualized_volatility": metrics["annualized_volatility"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "trade_count": metrics["trade_count"],
        "win_rate": metrics["win_rate"],
        "avg_win": metrics["avg_win"],
        "avg_loss": metrics["avg_loss"],
        "profit_factor": metrics["profit_factor"],
        "best_trade": metrics["best_trade"],
        "worst_trade": metrics["worst_trade"],
        "avg_trade_return": metrics["avg_trade_return"],
        "avg_holding_bars": metrics["avg_holding_bars"],
        "exposure": metrics["exposure"],
        "status": "success",
    }


def _failed_row(
    config: BatchConfig,
    dataset_path: Path,
    params: dict[str, int],
    exc: Exception,
) -> dict[str, str]:
    status = "blocked_by_quality_gate" if isinstance(exc, QualityGateError) else "failed"
    return {
        "experiment_id": config.experiment_id,
        "dataset_path": str(dataset_path),
        "strategy_file": str(config.strategy_file),
        "params": json.dumps(params, sort_keys=True),
        "status": status,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
