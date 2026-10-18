"""Defines when an OHLCV bar becomes observable by a strategy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from backtest_engine.core.timeframes import parse_timeframe_minutes

BAR_TIME_SEMANTICS = "open_with_available_at"


def ensure_bar_availability(frame: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """Attach canonical open, close, and availability timestamps to minute bars."""
    minutes = availability_minutes(timeframe)
    # ``ts`` is the canonical bar-open timestamp. Recompute derived timestamps
    # after aggregation and gap repair instead of trusting copied source values.
    close_ts = pl.col("ts") + pl.duration(minutes=minutes)
    expressions = [
        pl.col("ts").alias("bar_open_ts"),
        close_ts.alias("bar_close_ts"),
        close_ts.alias("available_at"),
    ]
    if "is_final" not in frame.columns:
        expressions.append(pl.lit(True).alias("is_final"))
    return frame.with_columns(expressions)


def finalized_as_of(
    frame: pl.DataFrame,
    *,
    timeframe: str,
    source_as_of: datetime,
    source_delay_minutes: int,
    timestamp_timezone: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Remove bars that were not safely final at snapshot creation time."""
    prepared = ensure_bar_availability(frame, timeframe)
    aware_as_of = source_as_of.replace(tzinfo=UTC) if source_as_of.tzinfo is None else source_as_of
    local_as_of = aware_as_of.astimezone(ZoneInfo(timestamp_timezone)).replace(tzinfo=None)
    bar_timezone = prepared.schema["available_at"].time_zone
    if bar_timezone:
        local_as_of = aware_as_of.astimezone(ZoneInfo(bar_timezone))
    cutoff = local_as_of - timedelta(minutes=source_delay_minutes)
    finalized = prepared.with_columns(
        (pl.col("available_at") <= pl.lit(cutoff)).alias("is_final")
    ).filter(pl.col("is_final"))
    return finalized, {
        "bar_time_semantics": BAR_TIME_SEMANTICS,
        "source_as_of": aware_as_of.astimezone(UTC).isoformat(),
        "source_delay_minutes": source_delay_minutes,
        "timestamp_timezone": bar_timezone or timestamp_timezone,
        "finalized_through": cutoff.isoformat(),
        "dropped_unfinalized_bar_count": prepared.height - finalized.height,
    }


def source_as_of_from_payload(payload: Any, fallback: datetime | None = None) -> datetime:
    raw = payload.get("generated_at") if isinstance(payload, dict) else None
    if raw:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or UTC)
    return fallback or datetime.now(UTC)


def row_available_at(row: dict[str, Any]) -> datetime:
    value = row.get("available_at") or row.get("bar_close_ts") or row["ts"]
    if not isinstance(value, datetime):
        raise TypeError(f"bar availability timestamp must be datetime, got {value!r}")
    return value


def row_open_ts(row: dict[str, Any]) -> datetime:
    value = row.get("bar_open_ts") or row["ts"]
    if not isinstance(value, datetime):
        raise TypeError(f"bar open timestamp must be datetime, got {value!r}")
    return value


def row_decision_boundary(row: dict[str, Any], bar_minutes: int) -> datetime:
    """Use the nominal boundary of an inclusive millisecond close for scheduling.

    Raw availability remains unchanged. A decision at the following boundary
    observes the completed bar; this never admits a bar that closes after it.
    """
    available = row_available_at(row)
    boundary = row_open_ts(row) + timedelta(minutes=bar_minutes)
    return boundary if available == boundary - timedelta(milliseconds=1) else available


def availability_minutes(timeframe: str) -> int:
    if timeframe == "1d":
        return 24 * 60
    return parse_timeframe_minutes(timeframe)
