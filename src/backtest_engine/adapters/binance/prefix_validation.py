"""Validate finalized observations before extending a pinned local dataset."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.adapters.binance.download import request_bounds
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.core.timeframes import parse_timeframe_minutes


def prefix_bounds(
    dataset: Path, request: DatasetBuildRequest, metadata: dict[str, Any], end_ms: int
) -> tuple[int, int, int] | None:
    schema = pl.read_parquet_schema(dataset)
    required = {
        "ts",
        "available_at",
        "is_final",
        "symbol",
        "engine",
        "timeframe",
        "base_currency",
        "quote_currency",
        "settlement_currency",
    }
    if not required.issubset(schema):
        return None
    first_ms, _ = request_bounds(request.from_date, request.till_date)
    first = datetime.fromtimestamp(first_ms / 1000, UTC)
    end = datetime.fromtimestamp(end_ms / 1000, UTC)
    frame = pl.scan_parquet(dataset).filter(
        (pl.col("ts") >= first) & (pl.col("available_at") <= end)
    )
    step_ms = parse_timeframe_minutes(request.timeframe) * 60_000
    expected = {
        "symbol": request.security,
        "engine": request.engine,
        "timeframe": request.timeframe,
        "base_currency": metadata["baseAsset"],
        "quote_currency": metadata["quoteAsset"],
        "settlement_currency": metadata.get("marginAsset") or metadata["quoteAsset"],
    }
    invalid = [
        pl.col("is_final").is_null() | ~pl.col("is_final"),
        pl.col("available_at").is_null(),
        ~(
            (pl.col("available_at") - pl.col("ts"))
            .dt.total_milliseconds()
            .is_in([step_ms - 1, step_ms])
        ),
        (pl.col("ts").diff().dt.total_milliseconds() != step_ms).fill_null(False),
        *[pl.col(name).is_null() | (pl.col(name) != value) for name, value in expected.items()],
    ]
    if request.engine == "perpetual":
        marks = ("mark_open", "mark_high", "mark_low", "mark_price")
        if not set(marks).issubset(schema):
            return None
        invalid.extend(
            pl.col(name).is_null() | ~pl.col(name).is_finite() | (pl.col(name) <= 0)
            for name in marks
        )
        invalid.extend(
            [
                pl.col("mark_low") > pl.min_horizontal("mark_open", "mark_price"),
                pl.col("mark_high") < pl.max_horizontal("mark_open", "mark_price"),
            ]
        )
    values = (
        frame.select(
            pl.col("ts").min().alias("first"),
            pl.col("ts").max().alias("last"),
            pl.any_horizontal(invalid).fill_null(True).any().alias("invalid"),
        )
        .collect()
        .row(0, named=True)
    )
    if values["last"] is None or values["invalid"] or values["first"] != first:
        return None
    last_ms = int(values["last"].timestamp() * 1000)
    return first_ms, last_ms, last_ms + step_ms
