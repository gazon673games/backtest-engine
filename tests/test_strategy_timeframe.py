from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.bar_aggregation import aggregate_intraday_bars


def test_strategy_bars_are_aggregated_independently_from_source() -> None:
    timestamps = [datetime(2024, 1, 2, 10, 0) + timedelta(minutes=5 * i) for i in range(4)]
    frame = pl.DataFrame(
        {
            "ts": timestamps,
            "timeframe": ["5m"] * 4,
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 14.0, 13.0, 16.0],
            "low": [9.0, 8.0, 11.0, 10.0],
            "close": [10.5, 12.0, 12.5, 15.0],
            "volume": [1.0, 2.0, 3.0, 4.0],
            "symbol": ["CRH4"] * 4,
        }
    )

    result = aggregate_intraday_bars(frame, "10m")

    assert result.get_column("timeframe").to_list() == ["10m", "10m"]
    assert result.select("open", "high", "low", "close", "volume").rows() == [
        (10.0, 14.0, 8.0, 12.0, 3.0),
        (12.0, 16.0, 10.0, 15.0, 7.0),
    ]


def test_strategy_timeframe_cannot_be_finer_than_approved_source() -> None:
    frame = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 2, 10, 0)],
            "timeframe": ["5m"],
            "open": [10.0],
            "high": [10.0],
            "low": [10.0],
            "close": [10.0],
        }
    )

    with pytest.raises(DatasetValidationError, match="finer than source"):
        aggregate_intraday_bars(frame, "1m")
