from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from backtest_engine.data.bar_aggregation import aggregate_intraday_bars
from backtest_engine.data.bar_availability import finalized_as_of


def test_finality_cutoff_excludes_bar_inside_source_delay_window() -> None:
    frame = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 2, 10, 0), datetime(2024, 1, 2, 10, 5)],
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
        }
    )

    result, audit = finalized_as_of(
        frame,
        timeframe="5m",
        source_as_of=datetime(2024, 1, 2, 7, 21, tzinfo=UTC),
        source_delay_minutes=15,
        timestamp_timezone="Europe/Moscow",
    )

    assert result.get_column("ts").to_list() == [datetime(2024, 1, 2, 10, 0)]
    assert audit["finalized_through"] == "2024-01-02T10:06:00"
    assert audit["dropped_unfinalized_bar_count"] == 1


def test_strategy_timeframe_recomputes_canonical_bar_boundaries() -> None:
    timestamps = [datetime(2024, 1, 2, 10, minute) for minute in range(5)]
    source = pl.DataFrame(
        {
            "ts": timestamps,
            "timeframe": ["1m"] * 5,
            "open": [1.0] * 5,
            "high": [1.0] * 5,
            "low": [1.0] * 5,
            "close": [1.0] * 5,
            "is_final": [True] * 5,
        }
    )

    result = aggregate_intraday_bars(source, "5m", source_timeframe="1m")

    assert result.row(0, named=True)["bar_open_ts"] == datetime(2024, 1, 2, 10, 0)
    assert result.row(0, named=True)["available_at"] == datetime(2024, 1, 2, 10, 5)
