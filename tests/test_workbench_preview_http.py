"""Chart requests must work with exchange timezones and return JSON on failure."""

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from backtest_engine.workbench import server
from backtest_engine.workbench.dataset_preview import dataset_preview_payload


@contextmanager
def preview_server(output_root):
    http = ThreadingHTTPServer(("127.0.0.1", 0), server.handler_class(output_root))
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", http.server_port, timeout=5)
    try:
        yield connection
    finally:
        connection.close()
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


@pytest.fixture
def chart_dataset(tmp_path, monkeypatch):
    def create(timezone):
        zone = ZoneInfo(timezone) if timezone else None
        start = datetime(2024, 1, 1, tzinfo=zone)
        times = [start + timedelta(minutes=5 * index) for index in range(60)]
        path = tmp_path / "bars.parquet"
        pl.DataFrame(
            {
                "ts": times,
                "open": range(60),
                "high": range(60),
                "low": range(60),
                "close": range(60),
                "volume": [1] * 60,
            }
        ).write_parquet(path)
        monkeypatch.setattr(
            "backtest_engine.workbench.dataset_preview.get_workbench_dataset",
            lambda _root, dataset_id: {
                "dataset_id": dataset_id,
                "dataset_path": str(path),
                "timeframe": "5m",
                "quality_status": "rejected",
            },
        )
        return times

    return create


@pytest.mark.parametrize("timezone", [None, "UTC", "Europe/Moscow"])
def test_http_preview_uses_dataset_wall_clock_dates(tmp_path, chart_dataset, timezone):
    times = chart_dataset(timezone)
    with preview_server(tmp_path) as connection:
        connection.request(
            "GET",
            "/api/datasets/prices/preview?from_ts=2024-01-01T01:00:00"
            "&till_ts=2024-01-01T02:00:00&sma_period=3&inflation_rate=8"
            "&inflation_base_ts=2024-01-01T00:00:00",
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200, payload
    assert payload["range_source_row_count"] == 13
    assert payload["rows"][0]["ts"] == str(times[12])
    assert payload["rows"][-1]["ts"] == str(times[24])
    assert payload["rows"][0]["chart_sma"] == 11
    assert payload["rows"][0]["chart_inflation"] > 100


def test_preview_converts_explicit_offsets_before_filtering_and_aggregation(
    tmp_path,
    chart_dataset,
):
    times = chart_dataset("UTC")
    payload = dataset_preview_payload(
        tmp_path,
        "prices",
        {
            "from_ts": ["2024-01-01T04:00:00+03:00"],
            "till_ts": ["2024-01-01T05:00:00+03:00"],
            "timeframe": ["60m"],
        },
    )
    assert payload["rows"][0]["ts"] == times[12]
    assert payload["rows"][0]["open"] == 12
    assert payload["rows"][0]["close"] == 23
    assert payload["rows"][0]["volume"] == 12


def test_preview_rejects_offset_without_dataset_timezone(tmp_path, chart_dataset):
    chart_dataset(None)
    with pytest.raises(ValueError, match="timestamps have no timezone"):
        dataset_preview_payload(tmp_path, "prices", {"from_ts": ["2024-01-01T00:00:00Z"]})


def test_preview_rejects_reversed_window(tmp_path, chart_dataset):
    chart_dataset("UTC")
    with preview_server(tmp_path) as connection:
        connection.request(
            "GET",
            "/api/datasets/prices/preview?from_ts=2024-01-02T00:00:00&till_ts=2024-01-01T00:00:00",
        )
        response = connection.getresponse()
        assert response.status == 400
        assert "from_ts" in json.loads(response.read())["error"]


def test_unexpected_preview_error_returns_json_and_server_survives(tmp_path, monkeypatch):
    def fail(*_args):
        raise pl.exceptions.SchemaError("invalid fixture timestamps")

    monkeypatch.setattr(server, "dataset_preview_payload", fail)
    with preview_server(tmp_path) as connection:
        connection.request("GET", "/api/datasets/broken/preview")
        response = connection.getresponse()
        assert response.status == 500
        assert response.getheader("Content-Type").startswith("application/json")
        assert json.loads(response.read())["error"] == (
            "/api/datasets/broken/preview: invalid fixture timestamps"
        )
        connection.request("GET", "/api/adapters")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["adapters"]
