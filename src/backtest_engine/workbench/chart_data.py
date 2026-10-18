"""Creates display-only bars and overlays for the Workbench market chart."""

from __future__ import annotations

import math
from datetime import datetime

import polars as pl

from backtest_engine.core.timeframes import parse_timeframe_minutes
from backtest_engine.data.bar_aggregation import OHLC_AGGREGATIONS

CHART_OVERLAY_COLUMNS = (
    "chart_sma",
    "chart_ema",
    "chart_bollinger_upper",
    "chart_bollinger_middle",
    "chart_bollinger_lower",
    "chart_inflation",
)
SECONDS_PER_YEAR = 365.2425 * 24 * 60 * 60


def prepare_chart_frame(
    frame: pl.DataFrame,
    *,
    source_timeframe: str | None,
    chart_timeframe: str | None,
    sma_period: int | None,
    inflation_rate: float | None,
    ema_period: int | None = None,
    bollinger_period: int | None = None,
    bollinger_stddev: float = 2.0,
    inflation_base_ts: datetime | None = None,
) -> tuple[pl.DataFrame, str | None]:
    target = chart_timeframe or source_timeframe
    result = aggregate_chart_bars(frame, source_timeframe, target)
    if sma_period is not None:
        if sma_period < 1:
            raise ValueError("sma_period must be positive")
        result = result.with_columns(pl.col("close").rolling_mean(sma_period).alias("chart_sma"))
    if ema_period is not None:
        if ema_period < 1:
            raise ValueError("ema_period must be positive")
        result = result.with_columns(
            pl.col("close")
            .ewm_mean(span=ema_period, adjust=False, min_samples=ema_period)
            .alias("chart_ema")
        )
    if bollinger_period is not None:
        if bollinger_period < 1:
            raise ValueError("bollinger_period must be positive")
        if not math.isfinite(bollinger_stddev) or bollinger_stddev <= 0:
            raise ValueError("bollinger_stddev must be finite and positive")
        middle = pl.col("close").rolling_mean(bollinger_period)
        deviation = pl.col("close").rolling_std(bollinger_period, ddof=0)
        result = result.with_columns(
            (middle + deviation * bollinger_stddev).alias("chart_bollinger_upper"),
            middle.alias("chart_bollinger_middle"),
            (middle - deviation * bollinger_stddev).alias("chart_bollinger_lower"),
        )
    if inflation_rate is not None:
        if not math.isfinite(inflation_rate) or inflation_rate <= -100.0:
            raise ValueError("inflation_rate must be finite and greater than -100")
        result = add_inflation_model(result, inflation_rate, inflation_base_ts)
    return result, target


def aggregate_chart_bars(
    frame: pl.DataFrame,
    source_timeframe: str | None,
    target_timeframe: str | None,
) -> pl.DataFrame:
    if not target_timeframe or not source_timeframe or target_timeframe == source_timeframe:
        return frame
    every = chart_interval(source_timeframe, target_timeframe)
    expressions = []
    for column in frame.columns:
        if column == "ts":
            continue
        operation = OHLC_AGGREGATIONS.get(column, "last")
        expressions.append(getattr(pl.col(column), operation)().alias(column))
    return (
        frame.sort("ts")
        .group_by_dynamic(
            "ts",
            every=every,
            period=every,
            closed="left",
            label="left",
            start_by="window",
        )
        .agg(expressions)
        .sort("ts")
    )


def chart_interval(source_timeframe: str, target_timeframe: str) -> str:
    try:
        source_minutes = parse_timeframe_minutes(source_timeframe)
    except ValueError as exc:
        raise ValueError(f"Unsupported source chart timeframe: {source_timeframe}") from exc
    if target_timeframe == "1d":
        return "1d"
    try:
        target_minutes = parse_timeframe_minutes(target_timeframe)
    except ValueError as exc:
        raise ValueError(f"Unsupported chart timeframe: {target_timeframe}") from exc
    if target_minutes < source_minutes or target_minutes % source_minutes != 0:
        raise ValueError(
            f"Chart timeframe {target_timeframe} must be equal to or a multiple of "
            f"source timeframe {source_timeframe}"
        )
    return f"{target_minutes}m"


def add_inflation_model(
    frame: pl.DataFrame,
    annual_rate: float,
    base_ts: datetime | None = None,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("chart_inflation"))
    start_ts = base_ts or frame.get_column("ts").min()
    annual_growth = math.log1p(annual_rate / 100.0)
    elapsed_seconds = (pl.col("ts") - pl.lit(start_ts)).dt.total_seconds()
    return frame.with_columns(
        (100.0 * (annual_growth * elapsed_seconds / SECONDS_PER_YEAR).exp()).alias(
            "chart_inflation"
        )
    )
