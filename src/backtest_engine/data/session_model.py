"""Evaluates trading sessions, holidays, short days, and row eligibility."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

from backtest_engine.core.exceptions import BacktestError

SESSION_FILTERS = frozenset({"all", "main", "evening"})
SHORT_DAY_POLICIES = frozenset({"allow", "skip"})


def session_updates_from_calendar(raw_calendar: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if "holiday_dates" in raw_calendar:
        updates["holiday_dates"] = [
            parse_date_value(value) for value in raw_calendar["holiday_dates"]
        ]
    if "holidays" in raw_calendar:
        updates["holiday_dates"] = [parse_date_value(value) for value in raw_calendar["holidays"]]
    if "short_day_dates" in raw_calendar:
        updates["short_day_dates"] = [
            parse_date_value(value) for value in raw_calendar["short_day_dates"]
        ]
    if "short_days" in raw_calendar:
        updates["short_day_dates"] = [
            parse_date_value(value) for value in raw_calendar["short_days"]
        ]
    for field_name in (
        "session_filter",
        "main_session_start",
        "main_session_end",
        "evening_session_start",
        "evening_session_end",
        "short_day_policy",
        "trading_calendar",
    ):
        if field_name in raw_calendar:
            updates[field_name] = raw_calendar[field_name]
    for field_name in (
        "main_session_start",
        "main_session_end",
        "evening_session_start",
        "evening_session_end",
    ):
        if field_name in updates and updates[field_name] is not None:
            updates[field_name] = parse_time_value(updates[field_name])
    return updates


def row_session_eligible(current_ts: datetime, session: Any) -> bool:
    if current_ts.date() in session.holiday_dates:
        return False
    if session.short_day_policy == "skip" and current_ts.date() in session.short_day_dates:
        return False
    label = session_label(current_ts, session)
    if session.session_filter == "all":
        return True
    return label == session.session_filter


def session_label(current_ts: datetime, session: Any) -> str:
    current_time = current_ts.time()
    if time_in_range(current_time, session.main_session_start, session.main_session_end):
        return "main"
    if session.evening_session_start is not None and session.evening_session_end is not None:
        if time_in_range(current_time, session.evening_session_start, session.evening_session_end):
            return "evening"
    return "off_session"


def time_in_range(value: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= value <= end
    return value >= start or value <= end


def parse_date_value(value: Any) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise BacktestError(f"Invalid date value: {value}. Expected YYYY-MM-DD") from exc


def parse_time_value(value: Any) -> time:
    if isinstance(value, time):
        return value
    try:
        parsed = time.fromisoformat(str(value))
    except ValueError as exc:
        raise BacktestError(f"Invalid time value: {value}. Expected HH:MM") from exc
    if parsed.second != 0 or parsed.microsecond != 0:
        raise BacktestError(f"Invalid time value: {value}. Expected minute precision HH:MM")
    return parsed
