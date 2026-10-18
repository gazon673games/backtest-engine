"""Small loopback Workbench client shared by CLI and MCP; no implicit job retries."""

import json
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import ProxyHandler, Request, build_opener


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


class WorkbenchClient:
    def __init__(self, url: str, timeout: float = 120):
        parsed = urlparse(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Workbench URL must be a loopback HTTP origin, e.g. http://127.0.0.1:8765"
            )
        self.url, self.timeout = url.rstrip("/"), timeout
        self.opener = build_opener(ProxyHandler({}))

    def call(
        self, path: str, *, payload: dict | None = None, query: dict | None = None
    ) -> dict[str, Any]:
        if not path.startswith("/api/"):
            raise ValueError("Workbench client only calls /api endpoints")
        if query:
            query = {
                k: str(v).lower() if isinstance(v, bool) else v
                for k, v in query.items()
                if v is not None
            }
            path += "?" + urlencode(query)
        body = json.dumps(payload, default=str).encode() if payload is not None else None
        request = Request(self.url + path, data=body, headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
        except HTTPError as exc:
            error = exc.read(8192).decode("utf-8", errors="replace")
            raise ValueError(f"Workbench {exc.code} at {path}: {error}") from exc
        except (URLError, TimeoutError) as exc:
            raise ConnectionError(
                f"Workbench unavailable at {self.url}; "
                f"start the application or workbench-serve: {exc}"
            ) from exc
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("Workbench response exceeds 2 MiB; request fewer rows/columns")
        value = json_safe(json.loads(raw))
        if len(json.dumps(value).encode()) > 64 * 1024:
            return {
                "status": "response_too_large",
                "endpoint": path,
                "hint": "Request fewer rows/columns or use the artifact file reference.",
            }
        return value
