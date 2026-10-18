from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest
from test_sdk_portfolio_native import bars

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.bar_aggregation import aggregate_intraday_bars
from backtest_engine.portfolio.multi.sdk_history import SDKHistory


def hourly_frame():
    original = bars("A", 100)
    return pl.concat(
        [
            original.with_columns(
                [
                    pl.col(c) + timedelta(hours=4 * i)
                    for c in ("ts", "bar_open_ts", "bar_close_ts", "available_at")
                ]
            )
            for i in range(9)
        ]
    )


def test_shifted_signal_uses_complete_past_buckets_and_preserves_raw_times():
    raw = hourly_frame()
    before = raw.clone()
    history = SDKHistory(raw, SimpleNamespace(timeframe="M720"), offset_minutes=60)
    assert history.frame["ts"][0] == datetime(2025, 1, 1, 1, tzinfo=UTC)
    assert history.advance(datetime(2025, 1, 1, 12, tzinfo=UTC)) is None
    assert history.advance(datetime(2025, 1, 1, 13, tzinfo=UTC)) == 0
    assert raw.equals(before)
    assert history.frame["close"][0] == raw.filter(pl.col("ts").dt.hour() == 12)["close"][0]


def test_shifted_history_excludes_a_bucket_with_a_missing_source_hour():
    raw = hourly_frame().filter(pl.col("ts") != datetime(2025, 1, 1, 5, tzinfo=UTC))
    history = SDKHistory(raw, SimpleNamespace(timeframe="M720"), offset_minutes=60)
    assert history.frame["ts"][0] == datetime(2025, 1, 1, 13, tzinfo=UTC)
    assert history.advance(datetime(2025, 1, 1, 13, tzinfo=UTC)) is None


def test_mid_history_start_never_executes_a_pre_start_warmup_signal():
    history = SDKHistory(hourly_frame(), SimpleNamespace(timeframe="M720"), offset_minutes=60)
    assert history.advance(datetime(2025, 1, 1, 17, tzinfo=UTC)) is None
    # The 13:00 close remains usable for indicators. Trading waits for the 01:00 close.
    assert history.advance(datetime(2025, 1, 2, 1, tzinfo=UTC)) == 1


def test_delayed_replay_event_does_not_backfill_an_expired_signal():
    history = SDKHistory(hourly_frame(), SimpleNamespace(timeframe="M720"), offset_minutes=60)
    assert history.advance(datetime(2025, 1, 1, 13, tzinfo=UTC)) == 0
    assert history.advance(datetime(2025, 1, 2, 2, tzinfo=UTC)) is None


def test_shifted_coarse_source_can_form_the_same_shifted_signal_clock():
    financial = aggregate_intraday_bars(hourly_frame(), "240m", offset_minutes=60)
    history = SDKHistory(financial, SimpleNamespace(timeframe="M720"), offset_minutes=60)
    assert history.frame["ts"][0] == datetime(2025, 1, 1, 1, tzinfo=UTC)
    assert history.frame["volume"][0] == 120000


@pytest.mark.parametrize("offset", [-1, 720, 1.5, True])
def test_invalid_signal_offset_is_rejected(offset):
    with pytest.raises(DatasetValidationError, match="offset_minutes"):
        aggregate_intraday_bars(hourly_frame(), "720m", offset_minutes=offset)


def test_shift_requiring_finer_data_is_rejected():
    financial = aggregate_intraday_bars(hourly_frame(), "240m")
    with pytest.raises(DatasetValidationError, match="align"):
        aggregate_intraday_bars(financial, "720m", offset_minutes=60)
