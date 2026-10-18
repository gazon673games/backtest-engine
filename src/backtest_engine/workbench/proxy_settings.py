"""Read and update persistent adapter proxy settings for Workbench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backtest_engine.adapters.proxy import (
    AdapterProxySettings,
    load_adapter_proxy_settings,
    save_adapter_proxy_settings,
)
from backtest_engine.adapters.registry import list_adapters


def proxy_settings_path(output_root: Path) -> Path:
    return output_root.parent / "proxy-settings.json"


def proxy_settings_payload(output_root: Path) -> dict[str, Any]:
    settings = load_adapter_proxy_settings(proxy_settings_path(output_root))
    return {
        "proxy": settings.model_dump(),
        "adapters": [{"id": adapter.id, "name": adapter.name} for adapter in list_adapters()],
    }


def save_proxy_settings_payload(
    output_root: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    known_ids = {adapter.id for adapter in list_adapters()}
    settings = AdapterProxySettings.model_validate(payload)
    unknown = sorted(set(settings.adapter_ids) - known_ids)
    if unknown:
        raise ValueError(f"unknown proxy adapter ids: {', '.join(unknown)}")
    if settings.all_adapters:
        settings.adapter_ids = []
    save_adapter_proxy_settings(proxy_settings_path(output_root), settings)
    return {"status": "completed", **proxy_settings_payload(output_root)}
