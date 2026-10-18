"""Composes the event-driven intraday domain into one reproducible run."""

from __future__ import annotations

from typing import Any

import polars as pl

from backtest_engine.accounting.ledger_frames import ledger_frame_from_snapshots
from backtest_engine.data.asset_schema import normalize_intraday_asset_frame
from backtest_engine.data.dataset import load_dataset
from backtest_engine.intraday.engine import validate_intraday_dataset
from backtest_engine.intraday.event.config import (
    INTRADAY_EVENT_CATALOG_ENFORCEMENTS,
    INTRADAY_EVENT_CONTRACT_ROUNDING,
    INTRADAY_EVENT_EXECUTION_ENGINES,
    INTRADAY_EVENT_INTRABAR_AMBIGUITY_POLICIES,
    INTRADAY_EVENT_MISSING_DECISION_POLICIES,
    INTRADAY_EVENT_ORDER_TYPES,
    INTRADAY_EVENT_PARTIAL_FILL_POLICIES,
    INTRADAY_EVENT_RISK_POLICIES,
    INTRADAY_EVENT_SCHEDULES,
    INTRADAY_EVENT_SESSION_FILTERS,
    INTRADAY_EVENT_SHORT_DAY_POLICIES,
    INTRADAY_EVENT_SIZING_MODES,
    INTRADAY_EVENT_TIMINGS,
    IntradayEventBacktestConfig,
    IntradayEventBacktestResult,
    IntradayEventDatasetConfig,
    IntradayEventExecutionConfig,
    IntradayEventInstrumentConfig,
    IntradayEventInstrumentSpec,
    IntradayEventOutputConfig,
    IntradayEventPortfolioConfig,
    IntradayEventSessionConfig,
    IntradayEventStrategyConfig,
)
from backtest_engine.intraday.event.data import (
    evaluate_intraday_event_catalog_approval,
    evaluate_intraday_event_quality_approval,
    resolve_event_config,
)
from backtest_engine.intraday.event.loop import EventProgressCallback, build_intraday_event_events
from backtest_engine.intraday.event.metrics import calculate_intraday_event_metrics
from backtest_engine.intraday.event.outputs import save_intraday_event_outputs
from backtest_engine.intraday.event.quality import enforce_runtime_decision_coverage
from backtest_engine.intraday.event.results import (
    build_intraday_event_decision_actions,
    build_intraday_event_equity,
)
from backtest_engine.intraday.event.timeframe import prepare_strategy_bars
from backtest_engine.intraday.event.trades import attribute_intraday_event_trades
from backtest_engine.intraday.validation import enforce_causal_bar_timestamps

__all__ = [
    "INTRADAY_EVENT_CATALOG_ENFORCEMENTS",
    "INTRADAY_EVENT_CONTRACT_ROUNDING",
    "INTRADAY_EVENT_EXECUTION_ENGINES",
    "INTRADAY_EVENT_INTRABAR_AMBIGUITY_POLICIES",
    "INTRADAY_EVENT_MISSING_DECISION_POLICIES",
    "INTRADAY_EVENT_ORDER_TYPES",
    "INTRADAY_EVENT_PARTIAL_FILL_POLICIES",
    "INTRADAY_EVENT_RISK_POLICIES",
    "INTRADAY_EVENT_SCHEDULES",
    "INTRADAY_EVENT_SESSION_FILTERS",
    "INTRADAY_EVENT_SHORT_DAY_POLICIES",
    "INTRADAY_EVENT_SIZING_MODES",
    "INTRADAY_EVENT_TIMINGS",
    "IntradayEventBacktestConfig",
    "IntradayEventBacktestResult",
    "IntradayEventDatasetConfig",
    "IntradayEventExecutionConfig",
    "IntradayEventInstrumentConfig",
    "IntradayEventInstrumentSpec",
    "IntradayEventOutputConfig",
    "IntradayEventPortfolioConfig",
    "IntradayEventSessionConfig",
    "IntradayEventStrategyConfig",
    "attribute_intraday_event_trades",
    "build_intraday_event_decision_actions",
    "build_intraday_event_equity",
    "build_intraday_event_events",
    "calculate_intraday_event_metrics",
    "evaluate_intraday_event_catalog_approval",
    "evaluate_intraday_event_quality_approval",
    "run_intraday_event_backtest",
    "run_intraday_event_backtest_frame",
    "save_intraday_event_outputs",
]


def run_intraday_event_backtest(
    config: IntradayEventBacktestConfig,
    progress_callback: EventProgressCallback | None = None,
) -> IntradayEventBacktestResult:
    df = normalize_intraday_asset_frame(load_dataset(config.dataset.path))
    dataset_validation = validate_intraday_dataset(df, config.dataset.path)
    return run_intraday_event_backtest_frame(
        config,
        df,
        dataset_validation,
        progress_callback=progress_callback,
    )


def run_intraday_event_backtest_frame(
    config: IntradayEventBacktestConfig,
    df: pl.DataFrame,
    dataset_validation: dict[str, Any],
    progress_callback: EventProgressCallback | None = None,
) -> IntradayEventBacktestResult:
    enforce_causal_bar_timestamps(
        dataset_validation,
        allow_legacy=config.dataset.allow_legacy_bar_timestamps,
        label=str(config.dataset.path),
    )
    tolerance_unspecified = config.execution.decision_tolerance_minutes is None
    config = resolve_event_config(config, dataset_validation)
    catalog_approval = evaluate_intraday_event_catalog_approval(config)
    quality_approval = evaluate_intraday_event_quality_approval(config, dataset_validation)
    strategy_frame, strategy_validation = prepare_strategy_bars(
        config,
        df,
        dataset_validation,
    )
    if tolerance_unspecified and config.execution.schedule == "times":
        execution = config.execution.model_copy(
            update={"decision_tolerance_minutes": strategy_validation["bar_minutes"]}
        )
        config = config.model_copy(update={"execution": execution})
    enforce_runtime_decision_coverage(
        config,
        strategy_frame,
        int(strategy_validation["bar_minutes"]),
        quality_approval,
    )
    ledger_snapshots: list[dict[str, Any]] = []
    events = build_intraday_event_events(
        config,
        strategy_frame,
        strategy_validation,
        progress_callback=progress_callback,
        ledger_snapshots=ledger_snapshots,
    )
    decision_actions = build_intraday_event_decision_actions(events)
    equity = build_intraday_event_equity(events)
    trades = attribute_intraday_event_trades(events, dataset_validation)
    metrics = calculate_intraday_event_metrics(events, trades, config.execution)
    output_dir = save_intraday_event_outputs(
        config,
        strategy_validation,
        quality_approval,
        catalog_approval,
        events,
        decision_actions,
        equity,
        trades,
        ledger_frame_from_snapshots(ledger_snapshots),
        metrics,
    )
    return IntradayEventBacktestResult(
        output_dir=output_dir,
        metrics=metrics,
        dataset_validation=strategy_validation,
        quality_approval=quality_approval,
        catalog_approval=catalog_approval,
    )
