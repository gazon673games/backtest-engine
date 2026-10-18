from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError

import polars as pl
import pytest
from binance_fixture import FixtureTransport, kline, timestamp_ms

from backtest_engine.adapters.binance.archive import archive_klines, read_zip_klines
from backtest_engine.adapters.binance.cache import CachedTransport
from backtest_engine.adapters.binance.download import (
    canonical_timeframe,
    download_klines,
    request_timeframe,
)
from backtest_engine.adapters.binance.normalize import normalize_klines
from backtest_engine.adapters.binance.transport import BinanceHttpTransport
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.bar_aggregation import aggregate_intraday_bars


def zipped(rows: list[list]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("bars.csv", "\n".join(",".join(map(str, row)) for row in rows))
    return stream.getvalue()


@pytest.mark.parametrize(
    ("value", "native", "canonical"),
    [
        ("1h", "1h", "60m"),
        ("120m", "2h", "120m"),
        ("1d", "1d", "1440m"),
        ("1440m", "1d", "1440m"),
    ],
)
def test_native_timeframes_preserve_minute_runtime_contract(value, native, canonical):
    assert request_timeframe(value) == native
    assert canonical_timeframe(value) == canonical


def test_spot_archive_converts_microsecond_timestamps():
    row = kline()
    row[0] = timestamp_ms(2025, 1, 1) * 1000
    row[6] = row[0] + 59_999_999
    result = read_zip_klines(zipped([row]), "fixture")
    assert result[0][0] == timestamp_ms(2025, 1, 1)
    assert result[0][6] == timestamp_ms(2025, 1, 1) + 59_999


def test_funding_at_bar_open_is_counted_once_even_when_rates_cancel():
    from binance_fixture import exchange_info

    first = kline()
    second = kline()
    second[0] += 60_000
    second[6] += 60_000
    events = [
        {"fundingTime": first[0], "fundingRate": "0.001"},
        {"fundingTime": first[0] + 30_000, "fundingRate": "-0.001"},
        {"fundingTime": second[0], "fundingRate": "0.002"},
    ]
    frame = normalize_klines(
        product="perpetual",
        symbol="BTCUSDT",
        timeframe="1m",
        metadata=exchange_info("perpetual")["symbols"][0],
        klines=[first, second],
        funding=events,
        mark_klines=[first, second],
    )
    assert frame.get_column("funding_rate").to_list() == [0.0, 0.002]
    assert frame.get_column("funding_event").to_list() == [True, True]


def test_archive_checksums_and_offline_reuse(tmp_path):
    class ArchiveTransport:
        def __init__(self):
            self.raw = zipped([kline()])
            self.calls = 0

        def read_bytes(self, url, **_kwargs):
            self.calls += 1
            return (
                hashlib.sha256(self.raw).hexdigest().encode()
                if url.endswith("CHECKSUM")
                else self.raw
            )

    transport = ArchiveTransport()
    kwargs = dict(
        product="spot", symbol="BTCUSDT", timeframe="1m", month="2024-01", cache_root=tmp_path
    )
    first = archive_klines(transport, **kwargs, policy="prefer")
    second = archive_klines(transport, **kwargs, policy="only")
    assert first == second
    assert transport.calls == 2
    assert Path(first[1]["path"]).is_file()
    cached = tmp_path / "archives/spot/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    cached.write_bytes(b"corrupt")
    with pytest.raises(DatasetValidationError, match="checksum mismatch"):
        archive_klines(transport, **kwargs, policy="only")


def test_page_cache_preserves_raw_and_detects_corruption(tmp_path):
    source = FixtureTransport()
    params = {"symbol": "BTCUSDT", "endTime": timestamp_ms(2024, 1, 2)}
    cache = CachedTransport(source, tmp_path, "prefer")
    expected = cache.get("spot", "/api/v3/klines", params)
    retry = CachedTransport(source, tmp_path, "prefer")
    assert retry.get("spot", "/api/v3/klines", params) == expected
    assert len(source.calls) == 1
    raw_path = Path(retry.provenance()[0]["path"])
    raw_path.write_bytes(b"corrupt")
    with pytest.raises(DatasetValidationError, match="hash mismatch"):
        retry.get("spot", "/api/v3/klines", params)


def test_offline_cache_fails_closed_on_missing_page(tmp_path):
    with pytest.raises(DatasetValidationError, match="offline cache missing"):
        CachedTransport(FixtureTransport(), tmp_path, "only").get("spot", "/api/v3/klines")


def test_interrupted_pagination_resumes_from_verified_pages(tmp_path):
    start = timestamp_ms(2024, 1, 1)
    end = start + 1002 * 60_000

    class Pages:
        def __init__(self, fail=False):
            self.fail, self.calls = fail, []

        def get(self, _product, _path, params):
            self.calls.append(params["startTime"])
            if params["startTime"] > start and self.fail:
                raise DatasetValidationError("connection interrupted")
            indices = range(1000) if params["startTime"] == start else range(1000, 1002)
            result = []
            for index in indices:
                row = kline()
                row[0] = start + index * 60_000
                row[6] = row[0] + 59_999
                result.append(row)
            return result

    kwargs = dict(product="spot", symbol="BTCUSDT", timeframe="1m", start_ms=start, end_ms=end)
    with pytest.raises(DatasetValidationError, match="interrupted"):
        download_klines(CachedTransport(Pages(fail=True), tmp_path, "prefer"), **kwargs)
    resumed = Pages()
    rows = download_klines(CachedTransport(resumed, tmp_path, "prefer"), **kwargs)
    assert len(rows) == 1002
    assert resumed.calls == [start + 1000 * 60_000]


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("https://fixture", 429, "limited", {"Retry-After": "2"}, io.BytesIO(b"limited")),
        URLError("connection interrupted"),
    ],
)
def test_transport_retries_transient_errors_without_network(monkeypatch, error):
    from backtest_engine.adapters.binance import transport as module

    sleeps = []
    responses = [error, io.BytesIO(b'{"ok": true}')]

    def open_url(*_args, **_kwargs):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(module, "open_adapter_url", open_url)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    assert BinanceHttpTransport().get("spot", "/fixture") == {"ok": True}
    assert len(sleeps) == 1


def test_aggregation_keeps_funding_and_mark_extremes():
    from datetime import UTC, datetime, timedelta

    stamps = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(5)]
    frame = pl.DataFrame(
        {
            "ts": stamps,
            "timeframe": ["1m"] * 5,
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "mark_high": [100.0, 120.0, 105.0, 101.0, 102.0],
            "mark_low": [99.0, 98.0, 70.0, 99.0, 99.0],
            "funding_rate": [0.1, 0.0, 0.2, 0.0, 0.0],
            "funding_event": [True, False, True, False, False],
        }
    )
    result = aggregate_intraday_bars(frame, "5m").row(0, named=True)
    assert result["funding_rate"] == pytest.approx(0.3)
    assert result["funding_event"] is True
    assert result["mark_high"] == 120.0
    assert result["mark_low"] == 70.0
