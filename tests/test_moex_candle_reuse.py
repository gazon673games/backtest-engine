from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.candle_transport import enrich_candle_rows
from backtest_engine.adapters.moex.iss.candles import download_moex_candles


@pytest.fixture
def local_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DatasetBuildRequest:
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setenv("WORKBENCH_MOEX_RAW_ROOT", str(raw))
    return DatasetBuildRequest(
        adapter_id="moex",
        engine="futures",
        market="forts",
        board="RFUD",
        security="CRH4",
        from_date="2024-01-02",
        till_date="2024-01-04",
        timeframe="5m",
        cache_dir=tmp_path / "cache",
    )


def stub_network(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls = []

    def download(request: DatasetBuildRequest, on_rows=None, **_kwargs) -> list[dict]:
        calls.append((request.from_date, request.till_date))
        day = date.fromisoformat(request.from_date)
        rows = []
        while day.isoformat() <= request.till_date:
            rows.append({"begin": f"{day} 10:00:00", "open": 1, "close": 1, "high": 1, "low": 1})
            day += timedelta(days=1)
        rows = enrich_candle_rows(rows, request)
        if on_rows is not None:
            on_rows(rows)
        return rows

    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.security_download.download_network_candles",
        download,
    )
    return calls


def test_complete_cache_reuses_different_requested_dates_without_network(
    local_request: DatasetBuildRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = stub_network(monkeypatch)
    assert len(download_moex_candles(local_request)) == 3
    narrower = local_request.model_copy(
        update={"from_date": "2024-01-03", "till_date": "2024-01-03"}
    )
    assert len(download_moex_candles(narrower)) == 1
    assert (
        len(download_moex_candles(local_request.model_copy(update={"cache_policy": "offline"})))
        == 3
    )
    assert calls == [("2024-01-02", "2024-01-04")]


def test_only_missing_left_and_right_intervals_are_downloaded(
    local_request: DatasetBuildRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = stub_network(monkeypatch)
    download_moex_candles(local_request)
    wider = local_request.model_copy(update={"from_date": "2024-01-01", "till_date": "2024-01-06"})
    published = []
    rows = download_moex_candles(wider, on_rows=published.extend)
    assert calls == [
        ("2024-01-02", "2024-01-04"),
        ("2024-01-01", "2024-01-01"),
        ("2024-01-05", "2024-01-06"),
    ]
    assert [row["begin"] for row in rows] == [f"2024-01-0{day} 10:00:00" for day in range(1, 7)]
    assert published == rows, "streamed cache data must not duplicate candles"
    assert download_moex_candles(wider) == rows
    assert len(calls) == 3


@pytest.mark.parametrize(
    "second_from, missing",
    [
        ("2024-01-04", []),
        ("2024-01-05", [("2024-01-04", "2024-01-04")]),
    ],
)
def test_multiple_cached_intervals_are_combined_without_redownloading_overlap(
    local_request: DatasetBuildRequest,
    monkeypatch: pytest.MonkeyPatch,
    second_from: str,
    missing: list[tuple[str, str]],
) -> None:
    calls = stub_network(monkeypatch)
    download_moex_candles(local_request.model_copy(update={"till_date": "2024-01-03"}))
    download_moex_candles(
        local_request.model_copy(update={"from_date": second_from, "till_date": "2024-01-06"})
    )
    calls.clear()
    request = local_request.model_copy(update={"till_date": "2024-01-06"})
    assert len(download_moex_candles(request)) == 5
    assert calls == missing


@pytest.mark.parametrize("policy", ["refresh", "network"])
def test_explicit_refresh_still_bypasses_cache(
    local_request: DatasetBuildRequest,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    calls = stub_network(monkeypatch)
    download_moex_candles(local_request)
    download_moex_candles(local_request.model_copy(update={"cache_policy": policy}))
    assert len(calls) == 2
