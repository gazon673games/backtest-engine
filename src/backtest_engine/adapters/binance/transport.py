"""Small injectable transport for Binance public REST endpoints."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request

from backtest_engine.adapters.proxy import open_adapter_url
from backtest_engine.core.exceptions import DatasetValidationError

BINANCE_BASE_URLS = {
    "spot": "https://data-api.binance.vision",
    "perpetual": "https://fapi.binance.com",
}


class BinanceTransport(Protocol):
    def get(
        self,
        product: str,
        path: str,
        params: dict[str, str | int] | None = None,
    ) -> Any: ...


class BinanceHttpTransport:
    def __init__(self, timeout_seconds: float = 30.0, max_attempts: int = 5) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    def get(
        self,
        product: str,
        path: str,
        params: dict[str, str | int] | None = None,
    ) -> Any:
        try:
            base_url = BINANCE_BASE_URLS[product]
        except KeyError as exc:
            raise DatasetValidationError(f"Unsupported Binance product: {product}") from exc
        query = urlencode(params or {})
        url = f"{base_url}{path}" + (f"?{query}" if query else "")
        try:
            return json.loads(self.read_bytes(url).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"Invalid Binance JSON for {path}") from exc

    def read_bytes(self, url: str, *, missing_ok: bool = False) -> bytes | None:
        request = Request(url, headers={"User-Agent": "ohlcv-replay/1"})
        for attempt in range(self.max_attempts):
            try:
                with open_adapter_url("binance", request, timeout=self.timeout_seconds) as response:
                    return response.read()
            except HTTPError as exc:
                if exc.code == 404 and missing_ok:
                    return None
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                delay = retry_delay(exc.headers.get("Retry-After"), attempt)
                if (
                    exc.code not in {429, 500, 502, 503, 504}
                    or delay > 60
                    or attempt + 1 == self.max_attempts
                ):
                    raise DatasetValidationError(
                        f"Binance HTTP {exc.code}: {url}; retry_after={delay}s; {detail}"
                    ) from exc
            except (URLError, TimeoutError, ConnectionError) as exc:
                if attempt + 1 == self.max_attempts:
                    raise DatasetValidationError(f"Binance request failed: {url}: {exc}") from exc
                delay = min(2**attempt, 30)
            time.sleep(delay)
        raise DatasetValidationError(f"Binance max_attempts must be positive: {self.max_attempts}")


def retry_delay(value: str | None, attempt: int) -> float:
    try:
        return max(0.0, float(value)) if value else float(min(2**attempt, 30))
    except ValueError:
        return 60.0


def endpoint(product: str, name: str) -> str:
    prefix = "/api/v3" if product == "spot" else "/fapi/v1"
    return f"{prefix}/{name}"
