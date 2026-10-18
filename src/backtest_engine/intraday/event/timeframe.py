"""Prepares the strategy bar stream while preserving source-data provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.data.bar_aggregation import aggregate_intraday_bars
from backtest_engine.intraday.engine import validate_intraday_dataset
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig


def prepare_strategy_bars(
    config: IntradayEventBacktestConfig,
    frame: pl.DataFrame,
    source_validation: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    target = config.strategy.timeframe
    if target is None or target == source_validation["timeframe"]:
        return frame, source_validation

    strategy_frame = aggregate_intraday_bars(
        frame,
        target,
        source_timeframe=str(source_validation["timeframe"]),
    )
    validation = validate_intraday_dataset(strategy_frame, Path(config.dataset.path))
    validation.update(
        {
            "source_timeframe": source_validation["timeframe"],
            "source_bar_minutes": source_validation["bar_minutes"],
            "source_row_count": source_validation["row_count"],
            "strategy_timeframe": target,
        }
    )
    return strategy_frame, validation
