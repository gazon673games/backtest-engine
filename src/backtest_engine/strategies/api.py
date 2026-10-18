from __future__ import annotations

import polars as pl

from backtest_engine.core.exceptions import StrategyPluginError


class StrategyContext:
    """Small helper API exposed to trusted local Python strategy plugins."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def frame(self) -> pl.DataFrame:
        return self._df.clone()

    def series(self, column: str) -> pl.Series:
        if column not in self._df.columns:
            msg = f"Strategy requested missing column: {column}"
            raise StrategyPluginError(msg)
        return self._df.get_column(column)

    def sma(self, column: str, window: int) -> pl.Series:
        _validate_window(window, "sma")
        return self.series(column).rolling_mean(window_size=window, min_samples=window)

    def ema(self, column: str, window: int) -> pl.Series:
        _validate_window(window, "ema")
        return self.series(column).ewm_mean(
            span=float(window),
            adjust=False,
            min_samples=window,
        )

    def shift(self, series: pl.Series, periods: int = 1) -> pl.Series:
        _validate_series_length(series, self._df.height, "shift")
        return series.shift(periods)

    def long_when(self, condition: pl.Series) -> pl.Series:
        _validate_series_length(condition, self._df.height, "long_when")
        return condition.fill_null(False).cast(pl.Float64)

    def flat(self) -> pl.Series:
        return pl.Series("signal", [0.0] * self._df.height, dtype=pl.Float64)


def _validate_window(window: int, helper: str) -> None:
    if window <= 0:
        msg = f"{helper} window must be greater than 0"
        raise StrategyPluginError(msg)


def _validate_series_length(series: pl.Series, expected_length: int, helper: str) -> None:
    if len(series) != expected_length:
        msg = f"{helper} series length must equal dataset row count"
        raise StrategyPluginError(msg)
