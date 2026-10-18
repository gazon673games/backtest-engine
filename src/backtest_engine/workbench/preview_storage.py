"""Parquet projection and bounded caching for chart previews."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from backtest_engine.data.store import parquet_metadata_index
from backtest_engine.workbench.chart_data import CHART_OVERLAY_COLUMNS, prepare_chart_frame

PRICE_COLUMNS = ("ts", "open", "high", "low", "close", "volume")


def preview_timestamp(value: datetime | None, dtype: pl.DataType) -> datetime | None:
    """Date inputs are wall-clock times in the dataset's timestamp timezone."""
    if value is None:
        return None
    if not isinstance(dtype, pl.Datetime):
        raise ValueError("dataset preview requires Datetime timestamps")
    if dtype.time_zone:
        zone = ZoneInfo(dtype.time_zone)
        return value.replace(tzinfo=zone) if value.tzinfo is None else value.astimezone(zone)
    if value.tzinfo is not None:
        raise ValueError("dataset timestamps have no timezone; use preview dates without an offset")
    return value


@lru_cache(maxsize=32)
def cached_source_row_count(dataset_path: str, _modified_ns: int, _file_size: int) -> int:
    return int(parquet_metadata_index(Path(dataset_path))["num_rows"])


def load_source_frame(dataset_path: str) -> pl.DataFrame:
    return source_scan(dataset_path).collect().sort("ts")


def load_range_source_frame(
    dataset_path: str,
    from_ts: datetime | None,
    till_ts: datetime | None,
    warmup_minutes: int,
) -> pl.DataFrame:
    scan = source_scan(dataset_path)
    if from_ts is not None:
        scan = scan.filter(pl.col("ts") >= from_ts - timedelta(minutes=warmup_minutes))
    if till_ts is not None:
        scan = scan.filter(pl.col("ts") <= till_ts)
    return scan.collect().sort("ts")


@lru_cache(maxsize=8)
def cached_preview_frame(
    dataset_path: str,
    modified_ns: int,
    file_size: int,
    source_timeframe: str | None,
    chart_timeframe: str | None,
    sma_period: int | None,
    ema_period: int | None,
    bollinger_period: int | None,
    bollinger_stddev: float,
    inflation_rate: float | None,
    inflation_base_ts: datetime | None,
    from_ts: datetime | None,
    till_ts: datetime | None,
    warmup_minutes: int,
    max_points: int,
) -> tuple[pl.DataFrame, str | None, int, int]:
    if from_ts is None and till_ts is None:
        source = load_source_frame(dataset_path)
    else:
        source = load_range_source_frame(
            dataset_path,
            from_ts,
            till_ts,
            warmup_minutes,
        )
    prepared, resolved_timeframe = prepare_chart_frame(
        source,
        source_timeframe=source_timeframe,
        chart_timeframe=chart_timeframe,
        sma_period=sma_period,
        inflation_rate=inflation_rate,
        ema_period=ema_period,
        bollinger_period=bollinger_period,
        bollinger_stddev=bollinger_stddev,
        inflation_base_ts=inflation_base_ts,
    )
    ranged = filter_time_range(prepared, from_ts, till_ts)
    source_count = ranged.height
    stride = max(1, math.ceil(source_count / max_points))
    frame = downsample_ohlcv(ranged, stride) if stride > 1 else ranged
    return frame, resolved_timeframe, source_count, stride


def filter_time_range(
    frame: pl.DataFrame,
    from_ts: datetime | None,
    till_ts: datetime | None,
) -> pl.DataFrame:
    predicates = []
    if from_ts is not None:
        predicates.append(pl.col("ts") >= from_ts)
    if till_ts is not None:
        predicates.append(pl.col("ts") <= till_ts)
    return frame.filter(*predicates) if predicates else frame


def downsample_ohlcv(frame: pl.DataFrame, stride: int) -> pl.DataFrame:
    expressions = [
        pl.col("ts").first().alias("ts"),
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
    ]
    if "volume" in frame.columns:
        expressions.append(pl.col("volume").sum().alias("volume"))
    expressions.extend(
        pl.col(column).last().alias(column)
        for column in CHART_OVERLAY_COLUMNS
        if column in frame.columns
    )
    return (
        frame.with_row_index("_preview_index")
        .with_columns((pl.col("_preview_index") // stride).alias("_preview_bucket"))
        .group_by("_preview_bucket", maintain_order=True)
        .agg(expressions)
        .drop("_preview_bucket")
    )


def source_scan(dataset_path: str) -> pl.LazyFrame:
    path = Path(dataset_path)
    schema = pl.read_parquet_schema(path)
    columns = [column for column in PRICE_COLUMNS if column in schema]
    missing = [column for column in PRICE_COLUMNS[:5] if column not in columns]
    if missing:
        raise ValueError(f"dataset preview missing columns: {', '.join(missing)}")
    return pl.scan_parquet(path).select(columns)
