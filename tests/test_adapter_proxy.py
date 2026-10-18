from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from backtest_engine.adapters.binance.transport import BinanceHttpTransport
from backtest_engine.adapters.proxy import (
    PROXY_SETTINGS_ENV,
    AdapterProxySettings,
    load_adapter_proxy_settings,
    normalized_proxy_url,
    proxy_url_for,
    save_adapter_proxy_settings,
)
from backtest_engine.workbench.server import get_payload, initialize_proxy_settings, post_payload


class ResponseStub:
    def __enter__(self) -> ResponseStub:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        return json.dumps({"symbols": []}).encode()


def test_proxy_settings_select_one_adapter_and_normalize_default_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "proxy.json"
    settings = AdapterProxySettings(
        enabled=True,
        url="localhost:10808",
        adapter_ids=["binance"],
    )
    save_adapter_proxy_settings(path, settings)
    monkeypatch.setenv(PROXY_SETTINGS_ENV, str(path))

    assert load_adapter_proxy_settings().url == "localhost:10808"
    assert proxy_url_for("binance") == "http://localhost:10808"
    assert proxy_url_for("moex") is None


def test_proxy_settings_can_target_all_adapters(tmp_path: Path, monkeypatch: Any) -> None:
    path = tmp_path / "proxy.json"
    save_adapter_proxy_settings(
        path,
        AdapterProxySettings(enabled=True, all_adapters=True),
    )
    monkeypatch.setenv(PROXY_SETTINGS_ENV, str(path))

    assert proxy_url_for("binance") == "http://localhost:10808"
    assert proxy_url_for("moex") == "http://localhost:10808"


def test_enabled_proxy_requires_adapter_selection() -> None:
    with pytest.raises(ValidationError, match="select at least one adapter"):
        AdapterProxySettings(enabled=True)


@pytest.mark.parametrize("url", ["socks5://localhost:10808", "localhost", "localhost:70000"])
def test_proxy_address_rejects_unsupported_or_incomplete_values(url: str) -> None:
    with pytest.raises(ValueError):
        normalized_proxy_url(url)


def test_binance_transport_uses_selected_proxy(tmp_path: Path, monkeypatch: Any) -> None:
    path = tmp_path / "proxy.json"
    save_adapter_proxy_settings(
        path,
        AdapterProxySettings(enabled=True, adapter_ids=["binance"]),
    )
    monkeypatch.setenv(PROXY_SETTINGS_ENV, str(path))
    captured: dict[str, Any] = {}

    class OpenerStub:
        def open(self, request: Any, *, timeout: float) -> ResponseStub:
            captured.update(request=request, timeout=timeout)
            return ResponseStub()

    def build_opener(*handlers: Any) -> OpenerStub:
        captured["handlers"] = handlers
        return OpenerStub()

    monkeypatch.setattr("backtest_engine.adapters.proxy.urllib.request.build_opener", build_opener)

    payload = BinanceHttpTransport(timeout_seconds=4).get("spot", "/api/v3/exchangeInfo")

    assert payload == {"symbols": []}
    assert captured["timeout"] == 4
    assert captured["handlers"][0].proxies == {
        "http": "http://localhost:10808",
        "https": "http://localhost:10808",
    }


def test_workbench_proxy_api_persists_multiple_adapters(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    response = post_payload(
        output_root,
        "/api/settings/proxy",
        {
            "enabled": True,
            "url": "http://127.0.0.1:10808",
            "adapter_ids": ["binance", "moex"],
            "all_adapters": False,
        },
    )
    loaded = get_payload(output_root, "/api/settings/proxy", "")

    assert response["status"] == "completed"
    assert loaded["proxy"]["adapter_ids"] == ["binance", "moex"]
    assert {item["id"] for item in loaded["adapters"]} >= {"binance", "moex"}
    assert (tmp_path / "proxy-settings.json").is_file()


def test_server_exports_proxy_path_for_isolated_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PROXY_SETTINGS_ENV, raising=False)

    path = initialize_proxy_settings(tmp_path / "runs")

    assert Path(os.environ[PROXY_SETTINGS_ENV]) == path
    assert load_adapter_proxy_settings(path).url == "localhost:10808"
