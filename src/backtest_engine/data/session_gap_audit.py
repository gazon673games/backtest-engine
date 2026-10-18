from __future__ import annotations

from datetime import datetime, time
from typing import Any

import polars as pl

from backtest_engine.core.timeframes import parse_timeframe_minutes

SessionWindows = tuple[tuple[str, str, str], ...]


def session_gap_audit(
    frame: pl.DataFrame,
    timeframe: str,
    session_windows: SessionWindows,
) -> dict[str, Any]:
    minutes = audit_timeframe_minutes(timeframe)
    timestamps = frame.sort("ts").get_column("ts").to_list()
    gaps = []
    missing_bar_count = 0
    expected_closures = 0
    for previous, current in zip(timestamps, timestamps[1:], strict=False):
        previous_window = session_window_id(previous, session_windows)
        current_window = session_window_id(current, session_windows)
        same_window = (
            previous.date() == current.date()
            and previous_window is not None
            and previous_window == current_window
        )
        delta = (current - previous).total_seconds() / 60.0
        if same_window and delta > minutes * 1.5:
            missing = max(int(delta // minutes) - 1, 1)
            missing_bar_count += missing
            gaps.append(gap_payload(previous, current, delta, previous_window, missing))
        elif not same_window and delta > minutes * 1.5:
            expected_closures += 1
    expected_bar_count = frame.height + missing_bar_count
    return {
        "status": "approved" if not gaps else "warning",
        "timeframe": timeframe,
        "expected_minutes": minutes,
        "gap_count": len(gaps),
        "missing_bar_count": missing_bar_count,
        "gap_ratio": (missing_bar_count / expected_bar_count if expected_bar_count > 0 else 0.0),
        "expected_closure_count": expected_closures,
        "session_aware": True,
        "session_windows": session_window_dicts(session_windows),
        "gaps": gaps[:100],
    }


def same_session_window(
    previous: datetime,
    current: datetime,
    session_windows: SessionWindows,
) -> bool:
    if previous.date() != current.date():
        return False
    previous_window = session_window_id(previous, session_windows)
    return previous_window is not None and previous_window == session_window_id(
        current,
        session_windows,
    )


def session_window_id(value: datetime, windows: SessionWindows) -> str | None:
    current = value.time()
    for name, raw_start, raw_end in windows:
        if time_in_range(current, time.fromisoformat(raw_start), time.fromisoformat(raw_end)):
            return name
    return None


def time_in_range(value: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= value <= end
    return value >= start or value <= end


def session_window_dicts(windows: SessionWindows) -> list[dict[str, str]]:
    return [{"id": name, "start": start, "end": end} for name, start, end in windows]


def gap_payload(
    previous: datetime,
    current: datetime,
    delta: float,
    session_name: str,
    missing_bar_count: int,
) -> dict[str, Any]:
    return {
        "previous_ts": previous.isoformat(),
        "current_ts": current.isoformat(),
        "gap_minutes": delta,
        "session": session_name,
        "missing_bar_count": missing_bar_count,
    }


def audit_timeframe_minutes(timeframe: str) -> int:
    if timeframe == "1d":
        return 24 * 60
    return parse_timeframe_minutes(timeframe)
