"""Probe remote history without relying on a downloaded local dataset."""

import json
from datetime import UTC, datetime
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Barrier, Thread

import pytest
from binance_fixture import FixtureTransport, kline, timestamp_ms

from backtest_engine.adapters.binance.adapter import BinanceAdapter
from backtest_engine.adapters.binance.history import binance_history_range
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.workbench import api, server

NOW = datetime(2024, 1, 2, 12, 30, tzinfo=UTC)


def bar(opened):
    row = kline()
    row[0] = opened
    row[6] = opened + 3_600_000 - 1
    return row


class HistoryTransport:
    def __init__(self):
        self.calls = []

    def get(self, product, path, params=None):
        self.calls.append((product, path, params))
        mark = path.endswith("markPriceKlines")
        if "startTime" in params:
            first = timestamp_ms(2019, 12, 23, 11) if mark else timestamp_ms(2017, 8, 17, 4)
            return [bar(first)]
        closed = timestamp_ms(2024, 1, 2, 10 if mark else 11)
        return [bar(closed), bar(timestamp_ms(2024, 1, 2, 12))]


@pytest.mark.parametrize(
    ("product", "expected", "count"),
    [("spot", "2017-08-18", 2), ("perpetual", "2019-12-24", 4)],
)
def test_range_probes_first_and_latest_finalized_bars(product, expected, count):
    transport = HistoryTransport()
    result = binance_history_range(
        transport, engine=product, symbol=" btcusdt ", timeframe="60m", now=NOW
    )
    assert result.symbol == "BTCUSDT"
    assert result.timeframe == "60m"
    assert result.from_date.isoformat() == expected
    assert result.till_date.isoformat() == "2024-01-02"
    assert result.last_bar.hour == (10 if product == "perpetual" else 11)
    assert len(transport.calls) == count
    assert all(call[2]["limit"] <= 2 for call in transport.calls)
    assert all(
        call[2]["interval"] == ("1m" if "startTime" in call[2] else "1h")
        for call in transport.calls
    )
    if product == "perpetual":
        assert result.first_bar.isoformat() == "2019-12-23T11:00:00+00:00"
        assert "markPriceKlines" in result.components


def test_midnight_first_bar_keeps_the_first_date():
    class Midnight(HistoryTransport):
        def get(self, product, path, params=None):
            if "startTime" in params:
                return [bar(timestamp_ms(2020, 1, 1))]
            return super().get(product, path, params)

    result = binance_history_range(
        Midnight(), engine="spot", symbol="BTCUSDT", timeframe="60m", now=NOW
    )
    assert result.from_date.isoformat() == "2020-01-01"


def test_perpetual_boundary_probes_run_concurrently():
    barrier = Barrier(4)

    class Concurrent(HistoryTransport):
        def get(self, product, path, params=None):
            barrier.wait(timeout=5)
            return super().get(product, path, params)

    result = binance_history_range(
        Concurrent(), engine="perpetual", symbol="BTCUSDT", timeframe="60m", now=NOW
    )
    assert result.from_date.isoformat() == "2019-12-24"


@pytest.mark.parametrize("response", [[], {}, [[1]], [[-1, 0, 0, 0, 0, 0, 1]]])
def test_invalid_or_empty_history_is_actionable(response):
    class Empty(HistoryTransport):
        def get(self, *args, **kwargs):
            return response

    with pytest.raises(DatasetValidationError, match="Binance"):
        binance_history_range(Empty(), engine="spot", symbol="BTCUSDT", timeframe="60m", now=NOW)


def test_no_finalized_candle_is_not_reported_as_available():
    class Future(HistoryTransport):
        def get(self, *args, **kwargs):
            return [bar(timestamp_ms(2024, 1, 2, 12))]

    with pytest.raises(DatasetValidationError, match="No finalized"):
        binance_history_range(Future(), engine="spot", symbol="BTCUSDT", timeframe="60m", now=NOW)


def test_invalid_identity_fails_before_requesting_binance():
    transport = HistoryTransport()
    for overrides in [{"engine": "inverse"}, {"symbol": "../BTC"}, {"timeframe": "7m"}]:
        with pytest.raises(DatasetValidationError):
            binance_history_range(
                transport,
                **({"engine": "spot", "symbol": "BTCUSDT", "timeframe": "60m"} | overrides),
            )
    assert transport.calls == []


def test_history_route_returns_dates_over_http(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "get_adapter", lambda adapter_id: BinanceAdapter(HistoryTransport()))
    http = ThreadingHTTPServer(("127.0.0.1", 0), server.handler_class(tmp_path / "runs"))
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", http.server_port, timeout=5)
    try:
        connection.request(
            "GET",
            "/api/adapters/binance/history-range?engine=perpetual&symbol=BTCUSDT&timeframe=60m",
        )
        response = connection.getresponse()
        assert response.status == 200
        payload = json.loads(response.read())
        assert payload["from_date"] == "2019-12-24"
        assert payload["timezone"] == "UTC"
        assert payload["start_policy"] == "first_complete_day"
        assert payload["engine"] == "perpetual"
        connection.request(
            "GET", "/api/adapters/binance/history-range?engine=inverse&symbol=BTCUSDT"
        )
        response = connection.getresponse()
        assert response.status == 400
        response.read()
    finally:
        connection.close()
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


def test_other_adapters_do_not_need_a_history_provider(monkeypatch):
    class Unsupported(BinanceAdapter):
        def info(self):
            info = super().info()
            info.capabilities.supports_history_range = False
            return info

    monkeypatch.setattr(api, "get_adapter", lambda adapter_id: Unsupported())
    with pytest.raises(ValueError, match="does not provide historical availability"):
        api.adapter_payload("other", "/history-range", {})


def test_all_pairs_are_visible_and_perpetual_engine_excludes_dated_contracts():
    class Catalog(FixtureTransport):
        def get(self, product, path, params=None):
            payload = super().get(product, path, params)
            template = payload["symbols"][0]
            payload["symbols"] = [
                dict(template, symbol=f"PAIR{index}USDT", contractType="PERPETUAL")
                for index in range(210)
            ] + [dict(template, symbol="BTCUSDT_240628", contractType="CURRENT_QUARTER")]
            return payload

    adapter = BinanceAdapter(Catalog())
    assert len(adapter.instruments(engine="spot", market="crypto", board="SPOT", query="")) == 211
    assert len(adapter.instruments(engine="perpetual", market="crypto", board="", query="")) == 210
    capabilities = adapter.info().capabilities
    assert capabilities.supports_history_range is True
    assert capabilities.default_rules_enforcement == "audit"
    assert capabilities.default_quality_approval_required is False
