from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from backtest_engine.strategies.intraday_api import IntradayStrategyContext


@pytest.mark.parametrize(
    "kind,operation,expected",
    [
        ("sma", "rolling_mean", [None, None, 7 / 3, 14 / 3]),
        ("ema", "ewm_mean", [None, None, 2.75, 5.375]),
    ],
)
def test_cached_indicators_match_hand_calculation_and_ignore_future_prices(
    monkeypatch: pytest.MonkeyPatch, kind: str, operation: str, expected: list[float | None]
) -> None:
    frame = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 2, 10) + timedelta(minutes=5 * i) for i in range(5)],
            "timeframe": ["5m"] * 5,
            "close": [1.0, 2.0, 4.0, 8.0, 999_999.0],
        }
    )
    original = getattr(pl.Series, operation)
    computations = []

    def record_computation(series: pl.Series, *args: object, **kwargs: object) -> pl.Series:
        computations.append(series.name)
        return original(series, *args, **kwargs)

    monkeypatch.setattr(pl.Series, operation, record_computation)
    cache = {}
    for index, value in enumerate(expected):
        ctx = IntradayStrategyContext(frame, end_index=index, indicator_cache=cache)
        actual = getattr(ctx, kind)("close", 3)
        if value is None:
            assert actual is None
        else:
            assert actual == pytest.approx(value)

    assert computations == ["close"]
