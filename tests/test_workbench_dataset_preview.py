from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.workbench.dataset_preview import dataset_preview_payload, downsample_ohlcv
from backtest_engine.workbench.preview_storage import (
    cached_preview_frame,
    load_range_source_frame,
    load_source_frame,
)


def test_preview_cache_only_wraps_bounded_output() -> None:
    assert not hasattr(load_source_frame, "cache_info")
    assert not hasattr(load_range_source_frame, "cache_info")
    assert cached_preview_frame.cache_info().maxsize == 8


def test_dataset_preview_is_sorted_downsampled_and_chunked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "bars.parquet"
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=index) for index in range(250)]
    pl.DataFrame(
        {
            "ts": list(reversed(timestamps)),
            "open": list(range(250)),
            "high": [value + 1 for value in range(250)],
            "low": [value - 1 for value in range(250)],
            "close": list(range(250)),
            "volume": [10] * 250,
            "ignored": ["large"] * 250,
        }
    ).write_parquet(path)
    monkeypatch.setattr(
        "backtest_engine.workbench.dataset_preview.get_workbench_dataset",
        lambda _root, dataset_id: {"dataset_id": dataset_id, "dataset_path": str(path)},
    )

    payload = dataset_preview_payload(
        tmp_path / "runs",
        "fixture",
        {"offset": ["20"], "limit": ["30"], "max_points": ["100"]},
    )

    assert payload["source_row_count"] == 250
    assert payload["preview_row_count"] == 84
    assert payload["sampling_stride"] == 3
    assert payload["next_offset"] == 50
    assert payload["done"] is False
    assert len(payload["rows"]) == 30
    assert payload["rows"] == sorted(payload["rows"], key=lambda row: row["ts"])
    assert "ignored" not in payload["rows"][0]


def test_dataset_preview_rejects_unbounded_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backtest_engine.workbench.dataset_preview.get_workbench_dataset",
        lambda _root, dataset_id: {"dataset_id": dataset_id, "dataset_path": str(tmp_path)},
    )

    with pytest.raises(ValueError, match="limit must be"):
        dataset_preview_payload(
            tmp_path / "runs",
            "fixture",
            {"limit": ["2001"]},
        )


def test_dataset_preview_preserves_ohlc_extremes_and_last_close() -> None:
    frame = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(minutes=index) for index in range(5)],
            "open": [10, 11, 12, 13, 14],
            "high": [11, 15, 13, 14, 17],
            "low": [9, 8, 11, 7, 13],
            "close": [11, 12, 13, 14, 16],
            "volume": [1, 2, 3, 4, 5],
        }
    )

    rows = downsample_ohlcv(frame, 3).to_dicts()

    assert rows == [
        {"ts": datetime(2024, 1, 1), "open": 10, "high": 15, "low": 8, "close": 13, "volume": 6},
        {
            "ts": datetime(2024, 1, 1, 0, 3),
            "open": 13,
            "high": 17,
            "low": 7,
            "close": 16,
            "volume": 9,
        },
    ]


def test_moving_averages_are_calculated_before_preview_downsampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "bars.parquet"
    start = datetime(2024, 1, 1)
    close = list(range(5_000))
    pl.DataFrame(
        {
            "ts": [start + timedelta(minutes=5 * index) for index in close],
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        }
    ).write_parquet(path)
    monkeypatch.setattr(
        "backtest_engine.workbench.dataset_preview.get_workbench_dataset",
        lambda _root, dataset_id: {
            "dataset_id": dataset_id,
            "dataset_path": str(path),
            "timeframe": "5m",
        },
    )

    payload = dataset_preview_payload(
        tmp_path / "runs",
        "fixture",
        {"max_points": ["100"], "sma_period": ["48"], "ema_period": ["48"]},
    )

    alpha = 2 / 49
    expected_ema = 0.0
    for value in range(1, 50):
        expected_ema = alpha * value + (1 - alpha) * expected_ema
    assert payload["sampling_stride"] == 50
    assert payload["rows"][0]["chart_sma"] == pytest.approx(sum(range(2, 50)) / 48)
    assert payload["rows"][0]["chart_ema"] == pytest.approx(expected_ema)


def test_dataset_preview_returns_exact_rows_for_visible_time_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "bars.parquet"
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(minutes=index) for index in range(250)]
    pl.DataFrame(
        {
            "ts": timestamps,
            "open": list(range(250)),
            "high": list(range(250)),
            "low": list(range(250)),
            "close": list(range(250)),
        }
    ).write_parquet(path)
    monkeypatch.setattr(
        "backtest_engine.workbench.dataset_preview.get_workbench_dataset",
        lambda _root, dataset_id: {
            "dataset_id": dataset_id,
            "dataset_path": str(path),
            "timeframe": "5m",
        },
    )

    payload = dataset_preview_payload(
        tmp_path / "runs",
        "fixture",
        {
            "from_ts": [timestamps[10].isoformat()],
            "till_ts": [timestamps[20].isoformat()],
            "max_points": ["100"],
        },
    )

    assert payload["source_row_count"] == 250
    assert payload["range_source_row_count"] == 11
    assert payload["preview_row_count"] == 11
    assert payload["sampling_stride"] == 1
    assert payload["exact"] is True
    assert payload["rows"][0]["ts"] == timestamps[10]
    assert payload["rows"][-1]["ts"] == timestamps[20]


def test_chart_timeframe_and_overlays_are_display_only(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "bars.parquet"
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=5 * index) for index in range(6)]
    pl.DataFrame(
        {
            "ts": timestamps,
            "open": [10, 11, 12, 13, 14, 15],
            "high": [11, 12, 13, 14, 15, 16],
            "low": [9, 10, 11, 12, 13, 14],
            "close": [10, 11, 12, 13, 14, 15],
            "volume": [1] * 6,
        }
    ).write_parquet(path)
    monkeypatch.setattr(
        "backtest_engine.workbench.dataset_preview.get_workbench_dataset",
        lambda _root, dataset_id: {
            "dataset_id": dataset_id,
            "dataset_path": str(path),
            "timeframe": "5m",
        },
    )

    payload = dataset_preview_payload(
        tmp_path / "runs",
        "fixture",
        {
            "timeframe": ["10m"],
            "sma_period": ["2"],
            "ema_period": ["2"],
            "bollinger_period": ["2"],
            "bollinger_stddev": ["2"],
            "inflation_rate": ["8"],
            "inflation_base_ts": ["2023-12-31T00:00:00"],
        },
    )

    assert payload["timeframe"] == "10m"
    assert payload["source_timeframe"] == "5m"
    assert len(payload["rows"]) == 3
    assert payload["rows"][0]["chart_sma"] is None
    assert payload["rows"][1]["chart_sma"] == pytest.approx(12.0)
    assert payload["rows"][0]["chart_ema"] is None
    assert payload["rows"][1]["chart_ema"] == pytest.approx(12.3333333333)
    assert payload["rows"][1]["chart_bollinger_upper"] == pytest.approx(14.0)
    assert payload["rows"][1]["chart_bollinger_middle"] == pytest.approx(12.0)
    assert payload["rows"][1]["chart_bollinger_lower"] == pytest.approx(10.0)
    assert payload["rows"][0]["chart_inflation"] == pytest.approx(100.0 * 1.08 ** (1 / 365.2425))
