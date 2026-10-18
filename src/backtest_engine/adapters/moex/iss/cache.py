from __future__ import annotations

import hashlib
import json
import os
import ssl
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import truststore

from backtest_engine.adapters.proxy import open_adapter_url, proxy_url_for

DEFAULT_CACHE_DIR = Path("data/workbench/cache/moex_iss")
VALID_CACHE_POLICIES = frozenset({"prefer", "refresh", "offline", "network"})
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def fetch_iss_json_cached(
    url: str,
    params: Mapping[str, str] | None = None,
    *,
    cache_policy: str = "prefer",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    policy = normalize_cache_policy(cache_policy)
    path = cache_path_for(url, params, cache_dir)
    if policy in {"prefer", "offline"} and path.exists():
        return read_cached_payload(path)
    if policy == "offline":
        raise FileNotFoundError(f"MOEX ISS cache miss in offline mode: {path}")
    try:
        payload = fetch_network_json(url, params)
    except Exception:
        if policy == "prefer" and path.exists():
            return read_cached_payload(path)
        raise
    if policy in {"prefer", "refresh"}:
        write_cached_payload(path, url, params, payload)
    return payload


def fetch_network_json(url: str, params: Mapping[str, str] | None = None) -> dict[str, Any]:
    full_url = url if not params else f"{url}?{urllib.parse.urlencode(params)}"
    if use_windows_native_transport(url):
        return fetch_windows_json(full_url, proxy_url=proxy_url_for("moex"))
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with open_adapter_url("moex", full_url, timeout=30, context=context) as response:
        payload = read_json_response(response)
    if not isinstance(payload, dict):
        raise ValueError("MOEX ISS response must be a JSON object")
    return payload


def use_windows_native_transport(url: str) -> bool:
    """Use SChannel for large candle pages; OpenSSL can stall on MOEX chunks on Windows."""
    return os.name == "nt" and url.rstrip("/").endswith("/candles.json")


def fetch_windows_json(full_url: str, *, proxy_url: str | None = None) -> dict[str, Any]:
    script = (
        "$ProgressPreference='SilentlyContinue';"
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);"
        "$params=@{UseBasicParsing=$true;Uri=$env:MOEX_ISS_URL;TimeoutSec=30};"
        "if($env:MOEX_ISS_PROXY){$params.Proxy=$env:MOEX_ISS_PROXY};"
        "$response=Invoke-WebRequest @params;"
        "[Console]::Out.Write($response.Content)"
    )
    env = {**os.environ, "MOEX_ISS_URL": full_url, "MOEX_ISS_PROXY": proxy_url or ""}
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        check=True,
        env=env,
        timeout=40,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if len(result.stdout) > MAX_RESPONSE_BYTES:
        raise ValueError("MOEX ISS response exceeds the 64 MiB safety limit")
    payload = json.loads(result.stdout.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("MOEX ISS response must be a JSON object")
    return payload


def read_json_response(response: Any) -> Any:
    """Stop once a complete JSON value arrives, even if a proxy keeps the chunk open."""
    if not hasattr(response, "read1") and not hasattr(response, "read"):
        if getattr(response, "chunked", False) and getattr(response, "fp", None) is not None:
            return read_chunked_json(response.fp)
        raise ValueError("MOEX ISS response does not expose a readable stream")
    content = bytearray()
    read_chunk = response.read1 if hasattr(response, "read1") else response.read
    while len(content) <= MAX_RESPONSE_BYTES:
        chunk = read_chunk(64 * 1024)
        if not chunk:
            break
        content.extend(chunk)
        try:
            return json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    if len(content) > MAX_RESPONSE_BYTES:
        raise ValueError("MOEX ISS response exceeds the 64 MiB safety limit")
    return json.loads(content.decode("utf-8"))


def read_chunked_json(stream: Any) -> Any:
    content = bytearray()
    while len(content) <= MAX_RESPONSE_BYTES:
        header = stream.readline(256)
        if not header:
            break
        try:
            chunk_size = int(header.split(b";", 1)[0].strip(), 16)
        except ValueError as exc:
            raise ValueError("MOEX ISS returned an invalid HTTP chunk") from exc
        if chunk_size == 0:
            break
        remaining = chunk_size
        read_part = stream.read1 if hasattr(stream, "read1") else stream.read
        while remaining > 0:
            part = read_part(min(remaining, 64 * 1024))
            if not part:
                raise ValueError("MOEX ISS HTTP chunk ended unexpectedly")
            content.extend(part)
            remaining -= len(part)
            try:
                return json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        stream.read(2)
    if len(content) > MAX_RESPONSE_BYTES:
        raise ValueError("MOEX ISS response exceeds the 64 MiB safety limit")
    return json.loads(content.decode("utf-8"))


def cache_path_for(
    url: str,
    params: Mapping[str, str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    key = cache_key(url, params)
    return cache_dir / f"{key}.json"


def cache_key(url: str, params: Mapping[str, str] | None = None) -> str:
    canonical = {
        "url": url,
        "params": sorted((params or {}).items()),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_cached_payload(path: Path) -> dict[str, Any]:
    cached = json.loads(path.read_text(encoding="utf-8"))
    payload = cached.get("payload") if isinstance(cached, dict) else None
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid MOEX ISS cache payload: {path}")
    return payload


def write_cached_payload(
    path: Path,
    url: str,
    params: Mapping[str, str] | None,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "url": url,
        "params": dict(params or {}),
        "fetched_at": datetime.now(UTC).isoformat(),
        "payload": payload,
    }
    path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def normalize_cache_policy(value: str) -> str:
    policy = value.strip().lower()
    if policy not in VALID_CACHE_POLICIES:
        choices = ", ".join(sorted(VALID_CACHE_POLICIES))
        raise ValueError(f"Invalid MOEX ISS cache policy: {value}. Expected one of: {choices}")
    return policy
