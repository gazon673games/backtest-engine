from __future__ import annotations

import json
from io import BytesIO
from typing import Any

from backtest_engine.adapters.moex.iss.cache import (
    fetch_network_json,
    read_json_response,
)
from backtest_engine.adapters.moex.iss.catalog import chain_payload


class ResponseStub:
    def __enter__(self) -> ResponseStub:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        return json.dumps({"status": "ok"}).encode()


def test_moex_transport_passes_system_trust_context(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def urlopen(_url: str, **kwargs: Any) -> ResponseStub:
        captured.update(kwargs)
        return ResponseStub()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    assert fetch_network_json("https://iss.moex.com/test") == {"status": "ok"}
    assert captured["context"].__class__.__module__.startswith("truststore")


def test_cny_rub_futures_root_has_human_readable_name() -> None:
    payload = chain_payload(
        "CR",
        [{"SECID": "CRM6", "SHORTNAME": "CNY/RUBF-6.26", "LASTTRADEDATE": "2026-06-18"}],
        "futures",
        "forts",
        "RFUD",
    )

    assert payload["display_name"] == "CNY/RUB - Chinese yuan / Russian ruble"


def test_complete_chunked_json_does_not_require_terminal_chunk() -> None:
    body = json.dumps({"items": [1, 2, 3]}).encode()

    class ChunkedResponse:
        chunked = True
        fp = BytesIO(f"{len(body):X}\r\n".encode() + body + b"\r\n")

    assert read_json_response(ChunkedResponse()) == {"items": [1, 2, 3]}


def test_http_response_reader_is_preferred_over_raw_chunk_stream() -> None:
    body = json.dumps({"items": ["decoded"]}).encode()

    class HttpResponse:
        chunked = True
        fp = BytesIO(b"invalid raw chunk framing")

        def read1(self, _size: int) -> bytes:
            return body

    assert read_json_response(HttpResponse()) == {"items": ["decoded"]}


def test_windows_candle_transport_uses_native_http(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.cache.use_windows_native_transport",
        lambda _url: True,
    )
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.cache.fetch_windows_json",
        lambda url, proxy_url=None: {"url": url, "proxy_url": proxy_url, "items": []},
    )

    payload = fetch_network_json("https://iss.moex.com/iss/test/candles.json", {"start": "5"})

    assert payload["url"].endswith("candles.json?start=5")
    assert payload["proxy_url"] is None


def test_windows_candle_transport_passes_selected_moex_proxy(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.cache.use_windows_native_transport",
        lambda _url: True,
    )
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.cache.proxy_url_for",
        lambda adapter_id: "http://localhost:10808" if adapter_id == "moex" else None,
    )
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.cache.fetch_windows_json",
        lambda _url, proxy_url=None: {"proxy_url": proxy_url},
    )

    payload = fetch_network_json("https://iss.moex.com/iss/test/candles.json")

    assert payload["proxy_url"] == "http://localhost:10808"
