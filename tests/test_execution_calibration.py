from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.execution.calibration import (
    calibrate_execution,
    calibration_updates,
)


def test_execution_calibration_fits_spread_impact_latency_and_participation(
    tmp_path: Path,
) -> None:
    start = datetime(2024, 1, 1)
    frame = pl.DataFrame(
        {
            "side": ["buy", "sell", "buy"],
            "arrival_mid": [100.0, 100.0, 100.0],
            "arrival_bid": [99.99, 99.99, 99.99],
            "arrival_ask": [100.01, 100.01, 100.01],
            "fill_price": [100.02, 99.97, 100.04],
            "order_quantity": [1.0, 2.0, 3.0],
            "market_volume": [10.0, 10.0, 10.0],
            "submitted_ts": [start, start, start],
            "filled_ts": [
                start + timedelta(seconds=10),
                start + timedelta(seconds=20),
                start + timedelta(seconds=30),
            ],
        }
    )

    calibration = calibrate_execution(frame)
    path = tmp_path / "calibration.json"
    calibration.write(path)
    updates = calibration_updates(path, bar_minutes=1)

    assert calibration.sample_count == 3
    assert calibration.median_spread_bps == pytest.approx(2.0)
    assert calibration.median_signed_slippage_bps == pytest.approx(3.0)
    assert calibration.median_market_impact_bps == pytest.approx(2.0)
    assert calibration.median_latency_ms == 20_000.0
    assert updates["latency_bars"] == 1
    assert updates["volume_participation_limit"] == pytest.approx(0.29)
