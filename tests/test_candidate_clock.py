"""A scheduled SDK sees inclusive closes at their nominal boundary, including midnight."""

from datetime import UTC, date, datetime, time, timedelta

import polars as pl

from backtest_engine.data.bar_aggregation import aggregate_intraday_bars
from backtest_engine.data.bar_availability import row_decision_boundary
from backtest_engine.intraday.event.config import (
    IntradayEventExecutionConfig,
    IntradayEventSessionConfig,
)
from backtest_engine.intraday.event.schedule import build_call_schedule


def test_inclusive_midnight_belongs_to_following_decision_date():
    start = datetime(2024, 1, 1, 23, 50, tzinfo=UTC)
    rows = [
        {
            "ts": start + timedelta(minutes=5 * i),
            "available_at": start + timedelta(minutes=5 * (i + 1), milliseconds=-1),
        }
        for i in range(5)
    ]
    schedule = build_call_schedule(
        rows,
        IntradayEventExecutionConfig(
            schedule="times",
            decision_times=[time(0)],
            decision_tolerance_minutes=0,
            from_date=date(2024, 1, 2),
            timing="signal_close_to_next_bar_open_to_close",
        ),
        IntradayEventSessionConfig(),
        5,
    )
    assert schedule.calls == {1: time(0)}
    assert schedule.missing_closes == {}
    assert rows[1]["available_at"] == datetime(2024, 1, 1, 23, 59, 59, 999000, tzinfo=UTC)
    assert row_decision_boundary(rows[1], 5) == datetime(2024, 1, 2, tzinfo=UTC)


def test_coarse_bars_preserve_late_availability_and_actual_first_open():
    opened = datetime(2018, 2, 9, 9, 58, 14, 800000, tzinfo=UTC)
    closed = opened + timedelta(minutes=5, milliseconds=-1)
    frame = pl.DataFrame(
        {
            "ts": [opened],
            "bar_open_ts": [opened],
            "bar_close_ts": [closed],
            "available_at": [closed],
            "timeframe": ["5m"],
            "close": [835.0],
        }
    )
    coarse = aggregate_intraday_bars(frame, "15m")
    assert coarse["ts"][0] == datetime(2018, 2, 9, 9, 45, tzinfo=UTC)
    assert coarse["bar_open_ts"][0] == opened
    assert coarse["available_at"][0] == closed
    assert coarse["bar_close_ts"][0] == closed
    assert frame["available_at"][0] == closed
