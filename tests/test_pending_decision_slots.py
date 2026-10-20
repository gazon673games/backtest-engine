"""Unfinished sessions defer later checks without invalidating earlier decisions."""

from datetime import date, datetime, time, timedelta

import pytest

from backtest_engine.intraday.event.config import (
    IntradayEventExecutionConfig,
    IntradayEventSessionConfig,
)
from backtest_engine.intraday.event.schedule import build_call_schedule

DAY = date(2024, 1, 2)


def rows_until(day, finish):
    opened = datetime.combine(day, time(9))
    end = datetime.combine(day, finish)
    rows = []
    while opened + timedelta(minutes=5) <= end:
        rows.append(
            {
                "ts": opened,
                "opened_at": opened,
                "available_at": opened + timedelta(minutes=5),
                "closed_at": opened + timedelta(minutes=5),
            }
        )
        opened += timedelta(minutes=5)
    return rows


@pytest.mark.parametrize("policy", ["carry", "close", "skip"])
@pytest.mark.parametrize("timing", ["signal_close_to_same_close", "signal_close_to_next_bar_close"])
@pytest.mark.parametrize(
    "finish, expected",
    [(time(13), []), (time(14), [time(13, 30)]), (time(15, 20), [time(13, 30), time(15)])],
)
def test_later_check_is_pending_in_partial_session(policy, timing, finish, expected):
    execution = IntradayEventExecutionConfig(
        schedule="times",
        decision_times=[time(13, 30), time(15)],
        missing_decision_policy=policy,
        timing=timing,
        allow_noncausal_same_close=timing == "signal_close_to_same_close",
        decision_tolerance_minutes=0,
    )
    schedule = build_call_schedule(
        rows_until(DAY, finish), execution, IntradayEventSessionConfig(), 5
    )
    assert list(schedule.calls.values()) == expected
    assert schedule.missing_closes == {}
    assert schedule.skipped_dates == set()


@pytest.mark.parametrize("policy", ["carry", "close", "skip"])
def test_history_check_does_not_make_fifteen_oclock_decision_early(policy):
    execution = IntradayEventExecutionConfig(
        schedule="times",
        decision_times=[time(15)],
        missing_decision_policy=policy,
    )
    schedule = build_call_schedule(
        rows_until(DAY, time(14)), execution, IntradayEventSessionConfig(), 5
    )
    assert schedule.calls == {}
    assert schedule.missing_closes == {}
    assert schedule.skipped_dates == set()


@pytest.mark.parametrize("policy", ["carry", "close", "skip"])
def test_missing_decision_policy_still_applies_to_elapsed_sessions(policy):
    execution = IntradayEventExecutionConfig(
        schedule="times",
        decision_times=[time(15)],
        missing_decision_policy=policy,
        decision_tolerance_minutes=0,
    )
    rows = rows_until(DAY, time(14)) + rows_until(DAY + timedelta(days=1), time(16))
    schedule = build_call_schedule(rows, execution, IntradayEventSessionConfig(), 5)
    assert len(schedule.calls) == 1
    assert (DAY in schedule.skipped_dates) == (policy == "skip")
    assert bool(schedule.missing_closes) == (policy == "close")
