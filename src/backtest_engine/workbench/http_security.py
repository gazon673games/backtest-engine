"""Server-side boundary for local browser writes and local agent requests."""

from collections.abc import Mapping


def validate_local_write(
    headers: Mapping[str, str], client: str, port: int, *, editing_source: bool
) -> None:
    host = headers.get("Host", "")
    origin = headers.get("Origin")
    local_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if client not in {"127.0.0.1", "::1"} or host not in local_hosts:
        raise ValueError("Requests require the local Workbench host")
    if origin is not None and origin != "http://" + host:
        raise ValueError("Requests require the local Workbench origin")
    if editing_source and origin is None:
        raise ValueError("Source editing requires the local Workbench origin")
    if origin is not None and headers.get("Content-Type", "").split(";")[0] != "application/json":
        raise ValueError("Browser requests require application/json")
