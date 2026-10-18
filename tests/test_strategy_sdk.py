from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.sdk import (
    StrategyParams,
    atr,
    bars_since_time,
    event_decision,
    feature_snapshot,
    flat_decision,
    higher_timeframe_sma,
    long_decision,
    resample_ohlcv,
    rolling_high,
    rolling_low,
    rsi,
    stop_limit_order,
    zscore,
)


def test_strategy_params_parse_typed_values_and_reject_unknowns() -> None:
    params = StrategyParams(
        {
            "window": "42",
            "threshold": "1.5",
            "enabled": "yes",
            "mode": "long",
        }
    )

    assert params.int("window", min_value=1) == 42
    assert params.float("threshold", min_value=0.0) == 1.5
    assert params.bool("enabled") is True
    assert params.str("mode", choices={"long", "short"}) == "long"
    params.reject_unused()


def test_strategy_params_reject_unknowns() -> None:
    params = StrategyParams({"window": "10", "surprise": "1"})

    assert params.int("window") == 10
    with pytest.raises(StrategyPluginError, match="Unknown strategy parameter"):
        params.reject_unused()


def test_strategy_sdk_indicators_use_only_visible_history() -> None:
    ctx = IntradayStrategyContext(intraday_frame(), end_index=3)

    assert rolling_high(ctx, window=3) == 13.0
    assert rolling_low(ctx, window=3) == 9.0
    assert zscore(ctx, "close", 3) == pytest.approx((12.0 - 11.0) / 0.816496580927726)
    assert atr(ctx, 3) == pytest.approx(2.0)
    assert rsi(ctx, "close", 3) == pytest.approx(100.0)
    assert bars_since_time(ctx, "10:05") == 3


def test_strategy_sdk_resamples_visible_history_only() -> None:
    ctx = IntradayStrategyContext(intraday_frame(), end_index=3)

    resampled = resample_ohlcv(ctx, "10m")

    assert resampled.get_column("close").to_list() == [10.0, 12.0]
    assert higher_timeframe_sma(ctx, "10m", "close", 2) == pytest.approx(11.0)


def test_strategy_sdk_decision_builders() -> None:
    features = feature_snapshot(sma=12.0, missing=None)

    assert long_decision(reason="go", features=features)["target"] == 1.0
    assert flat_decision(reason="stop")["target"] == 0.0
    assert event_decision(-1.0, order=stop_limit_order(10.0, 9.5)) == {
        "target": -1.0,
        "reason": None,
        "features": {},
        "order": {"type": "stop_limit", "stop_price": 10.0, "limit_price": 9.5},
    }


def intraday_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 2, 10, 0),
                datetime(2024, 1, 2, 10, 5),
                datetime(2024, 1, 2, 10, 10),
                datetime(2024, 1, 2, 10, 15),
                datetime(2024, 1, 2, 10, 20),
            ],
            "timeframe": ["5m"] * 5,
            "open": [9.0, 10.0, 11.0, 12.0, 1000.0],
            "high": [10.0, 11.0, 12.0, 13.0, 1000.0],
            "low": [8.0, 9.0, 10.0, 11.0, 1000.0],
            "close": [9.0, 10.0, 11.0, 12.0, 1000.0],
            "active_contract": ["CRH4"] * 5,
        }
    )
