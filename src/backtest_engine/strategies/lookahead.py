"""Mandatory prefix-differential gate for vectorized strategy plugins."""

from __future__ import annotations

import math

import polars as pl

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.core.models import StrategyConfig
from backtest_engine.strategies.plugins.loader import generate_plugin_signal


def enforce_vectorized_lookahead(frame: pl.DataFrame, config: StrategyConfig) -> None:
    if config.name != "plugin":
        return
    if config.strategy_file is None:
        raise StrategyPluginError("plugin strategy requires strategy_file")
    baseline = generate_plugin_signal(frame, config.strategy_file, config.strategy_params)
    for index in range(1, frame.height):
        prefix = frame.slice(0, index + 1)
        observed = generate_plugin_signal(
            prefix,
            config.strategy_file,
            config.strategy_params,
        )[-1]
        expected = baseline[index]
        if not math.isclose(
            observed,
            expected,
            rel_tol=0.0,
            abs_tol=config.lookahead_tolerance,
        ):
            raise StrategyPluginError(
                "Vectorized strategy failed mandatory lookahead audit at "
                f"index={index}: full_frame={expected}, prefix={observed}"
            )
