"""Standard daily benchmark controls and externally supplied plugins."""

from __future__ import annotations

import polars as pl

from backtest_engine.core.models import StrategyConfig
from backtest_engine.strategies.plugins.loader import generate_plugin_signal


def generate_signal(df: pl.DataFrame, config: StrategyConfig) -> list[float]:
    if config.name == "plugin":
        if config.strategy_file is None:
            raise ValueError("plugin strategy requires strategy_file")
        return generate_plugin_signal(df, config.strategy_file, config.strategy_params)
    if config.name == "buy_and_hold":
        return [1.0] * df.height
    closes = df.get_column("close")
    fast = closes.rolling_mean(window_size=config.fast_window)
    slow = closes.rolling_mean(window_size=config.slow_window)
    return (fast > slow).fill_null(False).cast(pl.Float64).to_list()
