from __future__ import annotations

import polars as pl

from backtest_engine.core.exceptions import StrategyPluginError


def crossed(
    values: pl.Series,
    threshold_series_or_value: pl.Series | float | int,
    lookback_bars: int,
    *,
    above: bool,
) -> bool:
    if lookback_bars <= 0:
        raise StrategyPluginError("crossed window must be greater than 0")
    if len(values) <= lookback_bars:
        return False
    current_value = values[-1]
    previous_value = values[-1 - lookback_bars]
    if current_value is None or previous_value is None:
        return False
    if isinstance(threshold_series_or_value, pl.Series):
        if len(threshold_series_or_value) <= lookback_bars:
            return False
        current_threshold = threshold_series_or_value[-1]
        previous_threshold = threshold_series_or_value[-1 - lookback_bars]
    else:
        current_threshold = threshold_series_or_value
        previous_threshold = threshold_series_or_value
    if current_threshold is None or previous_threshold is None:
        return False
    if above:
        return previous_value <= previous_threshold and current_value > current_threshold
    return previous_value >= previous_threshold and current_value < current_threshold
