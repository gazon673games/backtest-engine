from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.strategies.indicators import resample_ohlcv
from backtest_engine.strategies.intraday_api import IntradayStrategyContext


def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 2, 10) + timedelta(minutes=5 * i) for i in range(8)],
            "timeframe": ["5m"] * 8,
            "close": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 1e9],
            "open": [1.0] * 8,
            "high": [9.0] * 8,
            "low": [0.5] * 8,
            "volume": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )


@pytest.mark.parametrize("kind", ["sma", "ema"])
@pytest.mark.parametrize("window", [1, 2, 3, 9])
def test_cached_indicators_match_uncached_with_nulls_warmup_and_rewinds(kind: str, window: int):
    source = frame()
    cache = {}
    for index in [0, 1, 2, 3, 6, 4, 6]:
        cached = IntradayStrategyContext(source, end_index=index, indicator_cache=cache)
        uncached = IntradayStrategyContext(source, end_index=index)
        assert getattr(cached, kind)("close", window) == getattr(uncached, kind)("close", window)


@pytest.mark.parametrize("every", ["10m", "15m", "1d", "1w"])
def test_resampling_preserves_partial_buckets_null_closes_volume_and_rewinds(every: str):
    source = frame()
    cache = {}
    for index in [0, 1, 2, 3, 4, 6, 4, 6]:
        cached = IntradayStrategyContext(source, end_index=index, indicator_cache=cache)
        uncached = IntradayStrategyContext(source, end_index=index)
        assert_frame_equal(
            resample_ohlcv(cached, every), resample_ohlcv(uncached, every), check_exact=True
        )
    # At 10:00 only the first bar is visible; the future billion-priced close is unavailable.
    visible = resample_ohlcv(IntradayStrategyContext(source, end_index=0), every)
    assert visible["close"].to_list() == [1.0]
    assert visible["volume"].to_list() == [1.0]


def test_resampling_handles_unsorted_history_and_a_replaced_dataset():
    cache = {}
    for source in [frame(), frame().reverse(), frame().with_columns(pl.lit(20.0).alias("close"))]:
        for index in [2, 5]:
            cached = IntradayStrategyContext(source, end_index=index, indicator_cache=cache)
            uncached = IntradayStrategyContext(source, end_index=index)
            assert_frame_equal(resample_ohlcv(cached, "10m"), resample_ohlcv(uncached, "10m"))


@pytest.mark.parametrize("kind", ["sma", "ema"])
def test_indicator_cache_cannot_reuse_prices_from_a_different_dataset(kind: str):
    cache = {}
    original = frame().fill_null(3.0)
    replacement = original.with_columns(pl.lit(20.0).alias("close"))
    getattr(IntradayStrategyContext(original, end_index=5, indicator_cache=cache), kind)("close", 2)
    current = IntradayStrategyContext(replacement, end_index=5, indicator_cache=cache)
    assert getattr(current, kind)("close", 2) == 20.0


@pytest.mark.parametrize("every", ["1d", "1w", "1mo"])
def test_resampling_across_calendar_boundaries_keeps_future_highs_and_volume_hidden(every: str):
    timestamps = [datetime(2024, 2, 27, 10) + timedelta(days=i) for i in range(8)]
    source = (
        frame()
        .with_columns(pl.Series("ts", timestamps))
        .with_columns(
            pl.when(pl.int_range(pl.len()) == 7).then(1e12).otherwise(pl.col("high")).alias("high"),
            pl.when(pl.int_range(pl.len()) == 7)
            .then(1e12)
            .otherwise(pl.col("volume"))
            .alias("volume"),
        )
    )
    cache = {}
    for index in range(7):
        cached = IntradayStrategyContext(source, end_index=index, indicator_cache=cache)
        uncached = IntradayStrategyContext(source, end_index=index)
        actual = resample_ohlcv(cached, every)
        assert_frame_equal(actual, resample_ohlcv(uncached, every), check_exact=True)
        if actual.height:
            assert actual["high"].max() == 9.0
        assert actual["volume"].sum() <= 28.0


def test_cached_resampling_missing_columns_preserves_the_public_error():
    source = frame().drop("high")
    ctx = IntradayStrategyContext(source, end_index=0, indicator_cache={})
    with pytest.raises(StrategyPluginError, match="resample_ohlcv requires columns: high"):
        resample_ohlcv(ctx, "10m")


def test_resampling_reuses_completed_buckets_instead_of_scanning_every_prefix(monkeypatch):
    from backtest_engine.strategies import resampling

    count = 2000
    source = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 2) + timedelta(minutes=5 * i) for i in range(count)],
            "timeframe": ["5m"] * count,
            "close": [10.0] * count,
        }
    ).with_columns([pl.col("close").alias(column) for column in ("open", "high", "low")])
    heights = []
    original = resampling.aggregate_ohlcv

    def record_aggregation(history, every):
        heights.append(history.height)
        return original(history, every)

    monkeypatch.setattr(resampling, "aggregate_ohlcv", record_aggregation)
    cache = {}
    for index in [1, 9, 99, 999, 1998]:
        resample_ohlcv(
            IntradayStrategyContext(source, end_index=index, indicator_cache=cache), "30m"
        )
    assert heights[0] == count
    assert max(heights[1:]) <= 6
