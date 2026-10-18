"""Orchestrates simple intraday target-position backtests."""

from __future__ import annotations

from typing import Any

import polars as pl

from backtest_engine.accounting.intraday import (
    attribute_intraday_trades,
    build_intraday_equity_curve,
)
from backtest_engine.core.timeframes import parse_timeframe_minutes
from backtest_engine.data.asset_schema import normalize_intraday_asset_frame
from backtest_engine.data.dataset import load_dataset
from backtest_engine.intraday.config import (
    IntradayBacktestConfig,
    IntradayBacktestResult,
    IntradayDatasetConfig,
    IntradayExecutionConfig,
    IntradayOutputConfig,
    IntradayStrategyConfig,
)
from backtest_engine.intraday.decisions import (
    build_decision_points,
    resolve_intraday_config,
)
from backtest_engine.intraday.metrics import calculate_intraday_metrics
from backtest_engine.intraday.outputs import save_intraday_outputs
from backtest_engine.intraday.quality import (
    enforce_runtime_decision_coverage,
    evaluate_intraday_quality,
)
from backtest_engine.intraday.validation import (
    EXPECTED_INTRADAY_VALUES,
    REQUIRED_INTRADAY_COLUMNS,
    enforce_causal_bar_timestamps,
    validate_intraday_dataset,
)

__all__ = [
    "EXPECTED_INTRADAY_VALUES",
    "IntradayBacktestConfig",
    "IntradayBacktestResult",
    "IntradayDatasetConfig",
    "IntradayExecutionConfig",
    "IntradayOutputConfig",
    "IntradayStrategyConfig",
    "REQUIRED_INTRADAY_COLUMNS",
    "attribute_intraday_trades",
    "build_decision_points",
    "build_intraday_equity_curve",
    "calculate_intraday_metrics",
    "parse_timeframe_minutes",
    "run_intraday_backtest",
    "run_intraday_backtest_frame",
    "save_intraday_outputs",
    "validate_intraday_dataset",
]


def run_intraday_backtest(config: IntradayBacktestConfig) -> IntradayBacktestResult:
    df = normalize_intraday_asset_frame(load_dataset(config.dataset.path))
    dataset_validation = validate_intraday_dataset(df, config.dataset.path)
    return run_intraday_backtest_frame(config, df, dataset_validation)


def run_intraday_backtest_frame(
    config: IntradayBacktestConfig,
    df: pl.DataFrame,
    dataset_validation: dict[str, Any],
) -> IntradayBacktestResult:
    enforce_causal_bar_timestamps(
        dataset_validation,
        allow_legacy=config.dataset.allow_legacy_bar_timestamps,
        label=str(config.dataset.path),
    )
    quality_approval = evaluate_intraday_quality(
        config.dataset,
        dataset_validation,
        config.execution.from_date,
    )
    config = resolve_intraday_config(config, dataset_validation)
    decision_points, skipped_days = build_decision_points(df, config)
    enforce_runtime_decision_coverage(
        config.dataset,
        decision_points.height,
        skipped_days.height,
        quality_approval,
    )
    equity = build_intraday_equity_curve(decision_points, config.execution)
    trades = attribute_intraday_trades(equity, dataset_validation)
    metrics = calculate_intraday_metrics(
        equity,
        trades,
        decision_points,
        skipped_days,
        config.execution,
    )
    output_dir = save_intraday_outputs(
        config,
        dataset_validation,
        decision_points,
        equity,
        trades,
        skipped_days,
        metrics,
        quality_approval,
    )
    return IntradayBacktestResult(
        output_dir=output_dir,
        metrics=metrics,
        dataset_validation=dataset_validation,
        quality_approval=quality_approval,
    )
