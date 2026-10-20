from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date, datetime, time, timedelta
from typing import Any

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.bar_availability import (
    row_available_at,
    row_decision_boundary,
    row_open_ts,
)
from backtest_engine.data.session_model import row_session_eligible, session_label, time_in_range
from backtest_engine.execution.model import event_loop_stop
from backtest_engine.intraday.event.config import (
    IntradayEventExecutionConfig,
    IntradayEventSessionConfig,
)
from backtest_engine.intraday.event.state import CallSchedule

Row = dict[str, Any]
Rows = Sequence[Row]

row_session_eligible_for_event = row_session_eligible
session_label_for_event = session_label
time_in_range_for_event = time_in_range


def build_call_schedule(
    rows: Rows,
    execution: IntradayEventExecutionConfig,
    session: IntradayEventSessionConfig,
    bar_minutes: int,
) -> CallSchedule:
    if execution.schedule == "every-bar":
        return CallSchedule(
            calls={},
            missing_closes={},
            skipped_dates=set(),
            every_bar=True,
        )
    return timed_call_schedule(rows, execution, session, bar_minutes)


def timed_call_schedule(
    rows: Rows,
    execution: IntradayEventExecutionConfig,
    session: IntradayEventSessionConfig,
    bar_minutes: int,
) -> CallSchedule:
    tolerance_minutes = (
        bar_minutes
        if execution.decision_tolerance_minutes is None
        else execution.decision_tolerance_minutes
    )
    call_schedule: dict[int, time | None] = {}
    missing_closes: dict[int, list[time]] = {}
    skipped_dates: set[date] = set()
    executable_row_count = event_loop_stop(len(rows), execution.timing)
    for trading_date, indices in iter_trading_date_indices(rows, bar_minutes):
        if trading_date < execution.from_date:
            continue
        update_call_schedule_for_date(
            trading_date,
            rows,
            indices,
            execution,
            session,
            tolerance_minutes,
            executable_row_count,
            call_schedule,
            missing_closes,
            skipped_dates,
            bar_minutes,
        )
    return CallSchedule(
        calls=dict(sorted(call_schedule.items())),
        missing_closes=dict(sorted(missing_closes.items())),
        skipped_dates=skipped_dates,
    )


def update_call_schedule_for_date(
    trading_date: date,
    rows: Rows,
    indices: list[int],
    execution: IntradayEventExecutionConfig,
    session: IntradayEventSessionConfig,
    tolerance_minutes: int,
    executable_row_count: int,
    call_schedule: dict[int, time | None],
    missing_closes: dict[int, list[time]],
    skipped_dates: set[date],
    bar_minutes: int | None = None,
) -> None:
    if executable_row_count == 0:
        return
    horizon = row_available_at(rows[executable_row_count - 1])
    for decision_time in execution.decision_times:
        selected = select_decision_row(
            trading_date,
            rows,
            indices,
            decision_time,
            tolerance_minutes,
            session,
            bar_minutes,
        )
        if selected is not None and selected < executable_row_count:
            previous = call_schedule.get(selected)
            if previous is not None:
                raise DatasetValidationError(
                    "Decision times "
                    f"{previous.isoformat(timespec='minutes')} and "
                    f"{decision_time.isoformat(timespec='minutes')} map to the same bar; "
                    "choose a finer strategy timeframe"
                )
            call_schedule[selected] = decision_time
            continue
        decision_at = datetime.combine(trading_date, decision_time, horizon.tzinfo)
        # A future slot is pending, not a missing decision or a reason to skip this day.
        if decision_at > horizon:
            continue
        if execution.missing_decision_policy == "skip":
            skipped_dates.add(trading_date)
            continue
        if execution.missing_decision_policy != "close":
            continue
        action_row = select_missing_decision_action_row(
            trading_date,
            rows,
            indices,
            decision_time,
            session,
            bar_minutes,
        )
        if action_row is not None and action_row < executable_row_count:
            missing_closes.setdefault(action_row, []).append(decision_time)


def iter_trading_date_indices(
    rows: Rows, bar_minutes: int | None = None
) -> Iterator[tuple[date, list[int]]]:
    trading_date: date | None = None
    indices: list[int] = []
    for index in range(len(rows)):
        current_date = (
            decision_timestamp(rows[index], bar_minutes).date()
            if bar_minutes
            else row_open_ts(rows[index]).date()
        )
        if trading_date is not None and current_date != trading_date:
            yield trading_date, indices
            indices = []
        trading_date = current_date
        indices.append(index)
    if trading_date is not None:
        yield trading_date, indices


def rows_by_trading_date(rows: Rows) -> dict[date, list[int]]:
    """Compatibility helper; scheduling itself uses the streaming iterator above."""
    return dict(iter_trading_date_indices(rows))


def select_decision_row(
    trading_date: date,
    rows: Rows,
    indices: list[int],
    decision_time: time,
    tolerance_minutes: int,
    session: IntradayEventSessionConfig,
    bar_minutes: int | None = None,
) -> int | None:
    if not indices:
        return None
    sample_ts = row_available_at(rows[indices[0]])
    decision_dt = datetime.combine(trading_date, decision_time)
    if sample_ts.tzinfo is not None:
        decision_dt = decision_dt.replace(tzinfo=sample_ts.tzinfo)
    lower_bound = decision_dt - timedelta(minutes=tolerance_minutes)
    candidates = (
        index
        for index in indices
        if lower_bound <= decision_timestamp(rows[index], bar_minutes) <= decision_dt
        and row_session_eligible_for_event(row_open_ts(rows[index]), session)
    )
    return max(candidates, key=lambda index: row_available_at(rows[index]), default=None)


def select_missing_decision_action_row(
    trading_date: date,
    rows: Rows,
    indices: list[int],
    decision_time: time,
    session: IntradayEventSessionConfig,
    bar_minutes: int | None = None,
) -> int | None:
    if not indices:
        return None
    sample_ts = row_available_at(rows[indices[0]])
    decision_dt = datetime.combine(trading_date, decision_time)
    if sample_ts.tzinfo is not None:
        decision_dt = decision_dt.replace(tzinfo=sample_ts.tzinfo)
    eligible = [
        index
        for index in indices
        if row_session_eligible_for_event(row_open_ts(rows[index]), session)
    ]
    rows_before = [
        index for index in eligible if decision_timestamp(rows[index], bar_minutes) <= decision_dt
    ]
    if rows_before:
        return max(rows_before, key=lambda index: row_available_at(rows[index]))
    rows_after = [
        index for index in eligible if decision_timestamp(rows[index], bar_minutes) > decision_dt
    ]
    if rows_after:
        return min(rows_after, key=lambda index: row_available_at(rows[index]))
    return None


def decision_timestamp(row: Row, bar_minutes: int | None) -> datetime:
    return row_decision_boundary(row, bar_minutes) if bar_minutes else row_available_at(row)
