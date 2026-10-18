"""Validate observations before using them as statistical evidence."""

from datetime import date
from math import isfinite

import polars as pl

from backtest_engine.core.exceptions import ReportingError


def validate_return_events(events: pl.DataFrame, run_id: str) -> None:
    prefix = f"Research run {run_id}"
    if not {"ts", "net_return"}.issubset(events.columns):
        raise ReportingError(f"{prefix} requires ts and net_return columns")
    if events.is_empty():
        raise ReportingError(f"{prefix} has no return observations")
    timestamps = events.get_column("ts")
    if timestamps.dtype != pl.Date and not isinstance(timestamps.dtype, pl.Datetime):
        raise ReportingError(f"{prefix} ts must contain dates or datetimes")
    if timestamps.null_count() or timestamps.n_unique() != len(timestamps):
        raise ReportingError(f"{prefix} requires non-null unique timestamps")
    if not timestamps.is_sorted():
        raise ReportingError(f"{prefix} return timestamps must be increasing")
    for value in events.get_column("net_return"):
        try:
            valid = value is not None and not isinstance(value, bool) and isfinite(float(value))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ReportingError(f"{prefix} net_return must contain finite non-null values")


def completed_by(events: pl.DataFrame, boundary: date) -> pl.DataFrame:
    if "next_ts" not in events.columns:
        return events
    dtype = events.schema["next_ts"]
    if dtype == pl.Date:
        cutoff = pl.lit(boundary)
    elif isinstance(dtype, pl.Datetime):
        cutoff = pl.datetime(boundary.year, boundary.month, boundary.day, time_zone=dtype.time_zone)
    else:
        raise ReportingError("Research next_ts must contain dates or datetimes")
    if events.get_column("next_ts").null_count():
        raise ReportingError("Research next_ts must not contain nulls")
    if events.filter(pl.col("next_ts") < pl.col("ts")).height:
        raise ReportingError("Research next_ts must not precede ts")
    return events.filter(pl.col("next_ts") <= cutoff)
