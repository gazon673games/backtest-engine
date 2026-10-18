"""Completed SDK signal bars over the native common settlement timeline."""

from bisect import bisect_right
from datetime import timedelta

import polars as pl
from strategy_sdk.time import timeframe_minutes

from backtest_engine.data.bar_aggregation import aggregate_intraday_bars, single_timeframe
from backtest_engine.data.bar_availability import row_available_at


class SDKHistory:
    def __init__(self, frame: pl.DataFrame, requirements, offset_minutes: int = 0) -> None:
        minutes = timeframe_minutes(requirements.timeframe)
        source = single_timeframe(frame)
        source_minutes = timeframe_minutes(source)
        every = f"{minutes}m"
        observed = frame.group_by_dynamic(
            "ts", every=every, closed="left", start_by="window", offset=f"{offset_minutes}m"
        ).agg(
            pl.len().alias("observed_count"),
            pl.col("available_at").max().alias("observed_available_at"),
            pl.col("bar_close_ts").max().alias("observed_close"),
            pl.col("bar_open_ts").min().alias("observed_open"),
        )
        self.frame = (
            aggregate_intraday_bars(frame, every, offset_minutes=offset_minutes)
            .join(observed, on="ts")
            .filter(
                (pl.col("observed_count") == minutes // source_minutes)
                & (pl.col("observed_open") == pl.col("ts"))
                & pl.col("is_final")
            )
            .with_columns(
                pl.col("observed_available_at").alias("available_at"),
                pl.col("observed_close").alias("bar_close_ts"),
            )
            .drop("observed_count", "observed_available_at", "observed_close", "observed_open")
        )
        self.times = [row_available_at(r) for r in self.frame.select("available_at").to_dicts()]
        self.last = -1

    def advance(self, at):
        index = bisect_right(self.times, at) - 1
        if index <= self.last:
            return None
        self.last = index
        # Warmup is visible history, not a newly delivered signal. A fresh account
        # must wait for its next candle close instead of executing a pre-start bar.
        if at - self.times[index] > timedelta(milliseconds=1):
            return None
        return index
