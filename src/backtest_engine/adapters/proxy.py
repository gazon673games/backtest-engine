"""Adapter-scoped HTTP proxy configuration and transport helpers."""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_PROXY_URL = "localhost:10808"
PROXY_SETTINGS_ENV = "WORKBENCH_PROXY_SETTINGS_PATH"


class AdapterProxySettings(BaseModel):
    enabled: bool = False
    url: str = DEFAULT_PROXY_URL
    adapter_ids: list[str] = Field(default_factory=list)
    all_adapters: bool = False

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("proxy address must not be empty")
        normalized_proxy_url(cleaned)
        return cleaned

    @field_validator("adapter_ids")
    @classmethod
    def normalize_adapter_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def require_target_when_enabled(self) -> AdapterProxySettings:
        if self.enabled and not self.all_adapters and not self.adapter_ids:
            raise ValueError("select at least one adapter or all adapters")
        return self

    def applies_to(self, adapter_id: str) -> bool:
        return self.enabled and (self.all_adapters or adapter_id in self.adapter_ids)


def configure_proxy_settings_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    os.environ[PROXY_SETTINGS_ENV] = str(resolved)
    return resolved


def configured_proxy_settings_path() -> Path | None:
    value = os.environ.get(PROXY_SETTINGS_ENV)
    return Path(value).expanduser().resolve() if value else None


def load_adapter_proxy_settings(
    path: Path | None = None,
    *,
    defaults: AdapterProxySettings | None = None,
) -> AdapterProxySettings:
    target = path or configured_proxy_settings_path()
    if target is None or not target.exists():
        return defaults or AdapterProxySettings()
    payload = json.loads(target.read_text(encoding="utf-8"))
    return AdapterProxySettings.model_validate(payload)


def save_adapter_proxy_settings(path: Path, settings: AdapterProxySettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    serialized = json.dumps(settings.model_dump(), indent=2, ensure_ascii=False) + "\n"
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def proxy_url_for(adapter_id: str) -> str | None:
    settings = load_adapter_proxy_settings()
    return normalized_proxy_url(settings.url) if settings.applies_to(adapter_id) else None


def normalized_proxy_url(value: str) -> str:
    normalized = value.strip()
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("proxy scheme must be http or https")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("proxy port must be between 1 and 65535") from exc
    if not parsed.hostname or port is None:
        raise ValueError("proxy address must include a host and port")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("proxy address must not include a path, query, or fragment")
    return normalized.rstrip("/")


def open_adapter_url(
    adapter_id: str,
    target: str | urllib.request.Request,
    *,
    timeout: float,
    context: ssl.SSLContext | None = None,
) -> Any:
    proxy_url = proxy_url_for(adapter_id)
    if proxy_url is None:
        return urllib.request.urlopen(target, timeout=timeout, context=context)  # noqa: S310
    handlers: list[Any] = [urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})]
    if context is not None:
        handlers.append(urllib.request.HTTPSHandler(context=context))
    return urllib.request.build_opener(*handlers).open(target, timeout=timeout)
