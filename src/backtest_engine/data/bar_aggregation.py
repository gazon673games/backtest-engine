"""Builds transient OHLCV bars without changing the approved source dataset."""

from __future__ import annotations

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.timeframes import parse_timeframe_minutes
from backtest_engine.data.bar_availability import ensure_bar_availability

OHLC_AGGREGATIONS = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "value": "sum",
    "mark_open": "first",
    "mark_high": "max",
    "mark_low": "min",
    "mark_price": "last",
    "funding_rate": "sum",
    "funding_event": "any",
    "funding_price_estimated": "any",
}


def aggregate_intraday_bars(
    frame: pl.DataFrame,
    target_timeframe: str,
    *,
    source_timeframe: str | None = None,
    offset_minutes: int = 0,
) -> pl.DataFrame:
    """Aggregate a minute dataset to an equal or coarser minute timeframe."""

    source = source_timeframe or single_timeframe(frame)
    try:
        source_minutes = parse_timeframe_minutes(source)
        target_minutes = parse_timeframe_minutes(target_timeframe)
    except ValueError as exc:
        raise DatasetValidationError(str(exc)) from exc
    if target_minutes < source_minutes:
        raise DatasetValidationError(
            f"Strategy timeframe {target_timeframe} is finer than source timeframe {source}"
        )
    if target_minutes % source_minutes != 0:
        raise DatasetValidationError(
            f"Strategy timeframe {target_timeframe} must be a multiple of source timeframe {source}"
        )
    if type(offset_minutes) is not int or not 0 <= offset_minutes < target_minutes:
        raise DatasetValidationError("bar offset_minutes must be an integer within the target bar")
    if offset_minutes and not frame.is_empty():
        first = frame["ts"][0]
        source_origin = (first.hour * 60 + first.minute) % source_minutes
        if offset_minutes % source_minutes != source_origin:
            raise DatasetValidationError(
                "bar offset_minutes does not align with source bar openings"
            )
    if target_minutes == source_minutes:
        return frame

    expressions = []
    for column in frame.columns:
        if column in {"bar_open_ts", "bar_close_ts", "available_at"}:
            operation = "min" if column == "bar_open_ts" else "max"
            expressions.append(getattr(pl.col(column), operation)().alias("observed_" + column))
            continue
        if column == "ts":
            continue
        if column == "timeframe":
            expressions.append(pl.lit(target_timeframe).alias(column))
            continue
        if column == "funding_price_rate":
            expressions.append(
                pl.when(pl.col(column).is_null().any())
                .then(None)
                .otherwise(pl.col(column).sum())
                .alias(column)
            )
            continue
        operation = "all" if column == "is_final" else OHLC_AGGREGATIONS.get(column, "last")
        expressions.append(getattr(pl.col(column), operation)().alias(column))

    aggregated = (
        frame.sort("ts")
        .group_by_dynamic(
            "ts",
            every=f"{target_minutes}m",
            period=f"{target_minutes}m",
            closed="left",
            label="left",
            start_by="window",
            offset=f"{offset_minutes}m",
        )
        .agg(expressions)
        .sort("ts")
    )
    prepared = ensure_bar_availability(aggregated, target_timeframe)
    observed = [
        "observed_" + column
        for column in ("bar_open_ts", "bar_close_ts", "available_at")
        if column in frame.columns
    ]
    # Resumed or delayed source bars can become observable after the nominal
    # bucket boundary. Retain that delay and the first actual opening trade.
    corrections = []
    for column in ("bar_open_ts", "bar_close_ts", "available_at"):
        source_column = "observed_" + column
        if source_column in prepared.columns:
            corrections.append(pl.max_horizontal(column, source_column).alias(column))
    return prepared.with_columns(corrections).drop(observed)


def single_timeframe(frame: pl.DataFrame) -> str:
    if "timeframe" not in frame.columns:
        raise DatasetValidationError("Dataset has no timeframe column")
    values = frame.get_column("timeframe").drop_nulls().unique().to_list()
    if len(values) != 1:
        raise DatasetValidationError("Dataset timeframe must contain exactly one value")
    return str(values[0])
