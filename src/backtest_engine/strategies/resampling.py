"""Causal OHLCV aggregation with reuse of completed time buckets."""

from __future__ import annotations

import polars as pl

from backtest_engine.core.exceptions import StrategyPluginError


def aggregate_ohlcv(history: pl.DataFrame, every: str) -> pl.DataFrame:
    missing = {"ts", "open", "high", "low", "close"}.difference(history.columns)
    if missing:
        raise StrategyPluginError(f"resample_ohlcv requires columns: {', '.join(sorted(missing))}")
    aggregations = [
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
    ]
    for column in ("volume", "value"):
        if column in history.columns:
            aggregations.append(pl.col(column).sum().alias(column))
    return (
        history.sort("ts")
        .group_by_dynamic("ts", every=every, period=every, closed="left")
        .agg(aggregations)
    )


class ResampledHistory:
    def __init__(self, source: pl.DataFrame, every: str) -> None:
        self.source = source
        self.ordered = "ts" in source.columns and source["ts"].is_sorted()
        self.end_index = -1
        self.frame: pl.DataFrame | None = None
        self.completed: pl.DataFrame | None = None
        self.bucket_ends: pl.Series | None = None
        if self.ordered:
            self.completed = aggregate_ohlcv(source, every)
            self.bucket_ends = (
                source.select("ts")
                .with_row_index("row_index")
                .group_by_dynamic("ts", every=every, period=every, closed="left")
                .agg(pl.col("row_index").last())["row_index"]
            )

    def update(self, end_index: int, every: str) -> pl.DataFrame:
        if not self.ordered:
            return aggregate_ohlcv(self.source.head(end_index + 1), every).drop_nulls("close")
        if end_index != self.end_index or self.frame is None:
            # A precomputed bucket is visible only after its final input bar has been observed.
            count = self.bucket_ends.search_sorted(end_index, side="right")
            completed = self.completed.head(count)
            start = int(self.bucket_ends[count - 1]) + 1 if count else 0
            if start <= end_index:
                partial = aggregate_ohlcv(self.source.slice(start, end_index - start + 1), every)
                completed = pl.concat([completed, partial])
            self.frame = completed.drop_nulls("close")
            self.end_index = end_index
        return self.frame


IndicatorCache = dict[tuple[str, str, int], pl.DataFrame | pl.Series | ResampledHistory]
