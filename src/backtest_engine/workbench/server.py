"""Serves the local Workbench API and packaged static application."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from strategy_sdk.source_files import SourceConflict

from backtest_engine.adapters.proxy import (
    configure_proxy_settings_path,
    save_adapter_proxy_settings,
)
from backtest_engine.core.exceptions import BacktestError
from backtest_engine.workbench.agent_api import (
    get_agent_payload,
    post_agent_payload,
    prepare_native_experiment,
    submit_native_experiment,
)
from backtest_engine.workbench.api import (
    adapter_payload,
    adapters_payload,
    async_dataset_build_payload,
    async_run_payload,
    cancel_job_payload,
    compare_payload,
    dataset_build_live_preview_payload,
    dataset_build_payload,
    dataset_get_payload,
    dataset_list_payload,
    dataset_preview_payload,
    jobs_payload,
    parse_query,
    progress_payload,
    research_report_payload,
    settings_proxy_payload,
    settings_proxy_save_payload,
    strategy_get_payload,
    strategy_list_payload,
    strategy_metadata_payload,
    strategy_save_payload,
    sweep_submit_payload,
    validation_payload,
)
from backtest_engine.workbench.history import (
    append_workbench_history,
    failure_record,
    history_path_for,
    load_workbench_history,
)
from backtest_engine.workbench.http_security import validate_local_write
from backtest_engine.workbench.models import WorkbenchRunRequest
from backtest_engine.workbench.portfolio_api import (
    portfolio_list_payload,
    portfolio_run_payload,
    portfolio_save_payload,
)
from backtest_engine.workbench.proxy_settings import proxy_settings_path
from backtest_engine.workbench.report import build_workbench_report
from backtest_engine.workbench.runner import generated_run_id, safe_run_id
from backtest_engine.workbench.sdk_editing import source_open_payload, source_save_payload
from backtest_engine.workbench.settings import load_workbench_settings

_GET_HANDLERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "/api/jobs": jobs_payload,
    "/api/settings/proxy": settings_proxy_payload,
    "/api/datasets": dataset_list_payload,
    "/api/strategies": strategy_list_payload,
    "/api/portfolios": portfolio_list_payload,
}
_POST_HANDLERS: dict[str, Callable[[Path, dict[str, Any]], dict[str, Any]]] = {
    "/api/settings/proxy": settings_proxy_save_payload,
    "/api/datasets/build": dataset_build_payload,
    "/api/datasets/build/async": async_dataset_build_payload,
    "/api/strategies": strategy_save_payload,
    "/api/sdk-source/open": source_open_payload,
    "/api/sdk-source/save": source_save_payload,
    "/api/strategy-metadata": strategy_metadata_payload,
    "/api/runs/async": async_run_payload,
    "/api/portfolios": portfolio_save_payload,
    "/api/portfolio-runs/async": portfolio_run_payload,
    "/api/research/sweeps": sweep_submit_payload,
    "/api/research/protocols/prepare": prepare_native_experiment,
    "/api/research/protocols/submit": submit_native_experiment,
    "/api/compare": compare_payload,
    "/api/validation": validation_payload,
}

STATIC_ROOT = next(
    (
        path
        for path in (
            Path(os.environ["WORKBENCH_STATIC_ROOT"])
            if os.environ.get("WORKBENCH_STATIC_ROOT")
            else None,
            Path(__file__).with_name("static"),
            Path(__file__).resolve().parents[4] / "ui" / "web",
        )
        if path is not None and path.exists()
    ),
    Path(__file__).with_name("static"),
)


def serve_workbench(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    output_root: Path = Path("data/workbench/runs"),
) -> None:
    initialize_proxy_settings(output_root)
    server = ThreadingHTTPServer((host, port), handler_class(output_root))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        from backtest_engine.workbench.jobs import close_manager

        close_manager(output_root)


def handler_class(output_root: Path) -> type[BaseHTTPRequestHandler]:
    class WorkbenchHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                payload = get_payload(output_root, parsed.path, parsed.query)
            except FileNotFoundError as exc:
                self.write_error(HTTPStatus.NOT_FOUND, str(exc))
                return
            except (BacktestError, ValueError, SyntaxError, KeyError, TypeError) as exc:
                if parsed.path.startswith("/api/"):
                    self.write_error(HTTPStatus.BAD_REQUEST, str(exc))
                else:
                    self.write_static()
                return
            except Exception as exc:
                logging.getLogger(__name__).exception("GET %s failed", parsed.path)
                self.write_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{parsed.path}: {exc}")
                return
            self.write_json(payload)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                validate_local_write(
                    self.headers,
                    self.client_address[0],
                    self.server.server_port,
                    editing_source=parsed.path.startswith("/api/sdk-source/"),
                )
            except ValueError as exc:
                self.write_error(HTTPStatus.FORBIDDEN, str(exc))
                return
            try:
                payload = post_payload(output_root, parsed.path, self.read_json())
                self.write_json(payload)
            except SourceConflict as exc:
                self.write_error(HTTPStatus.CONFLICT, str(exc))
            except Exception as exc:
                self.record_failure(locals().get("payload", {}), exc)
                self.write_json({"status": "failed", "error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def write_static(self) -> None:
            relative = "index.html" if self.path in {"/", "/index.html"} else self.path.lstrip("/")
            path = (STATIC_ROOT / relative).resolve()
            if not path.is_relative_to(STATIC_ROOT.resolve()) or not path.is_file():
                self.write_error(HTTPStatus.NOT_FOUND, "file not found")
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 8 * 1024 * 1024:
                raise ValueError("Request exceeds 8 MiB")
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON payload must be an object")
            return payload

        def write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def write_error(self, status: HTTPStatus, message: str) -> None:
            self.write_json({"status": "failed", "error": message}, status)

        def record_failure(self, payload: dict[str, Any], exc: Exception) -> None:
            try:
                payload.setdefault("output_root", str(output_root))
                request = WorkbenchRunRequest.model_validate(payload)
                run_id = safe_run_id(request.run_id or generated_run_id(request))
                append_workbench_history(
                    history_path_for(output_root),
                    failure_record(request, run_id=run_id, error=str(exc)),
                )
            except Exception:
                return

        def log_message(self, *_args: Any) -> None:
            return

    return WorkbenchHandler


def get_payload(output_root: Path, path: str, query: str) -> dict[str, Any]:
    if path == "/api/health":
        return {"service": "ohlcv-replay-workbench", "status": "ready"}
    if path.startswith("/api/agent/"):
        return get_agent_payload(output_root, path, query)
    handler = _GET_HANDLERS.get(path)
    if handler is not None:
        return handler(output_root)
    if path == "/api/history":
        return load_workbench_history(history_path_for(output_root))
    if path == "/api/adapters":
        return adapters_payload()
    if path.startswith("/api/dataset-builds/") and path.endswith("/preview"):
        job_id = unquote(path.removeprefix("/api/dataset-builds/").removesuffix("/preview"))
        return dataset_build_live_preview_payload(
            output_root,
            job_id,
            parse_query(query),
        )
    if path.startswith("/api/datasets/") and path.endswith("/preview"):
        dataset_id = unquote(path.removeprefix("/api/datasets/").removesuffix("/preview"))
        return dataset_preview_payload(output_root, dataset_id, parse_query(query))
    if path.startswith("/api/datasets/"):
        dataset_id = unquote(path.removeprefix("/api/datasets/"))
        return dataset_get_payload(output_root, dataset_id)
    if path.startswith("/api/strategies/"):
        return strategy_get_payload(output_root, unquote(path.removeprefix("/api/strategies/")))
    if path.startswith("/api/progress/"):
        return progress_payload(output_root, path.removeprefix("/api/progress/"))
    if path.startswith("/api/research/"):
        sweep_id = unquote(path.removeprefix("/api/research/"))
        return research_report_payload(output_root, sweep_id)
    if path.startswith("/api/adapters/"):
        relative = path.removeprefix("/api/adapters/")
        adapter_id, separator, _resource = relative.partition("/")
        if not separator:
            raise ValueError("adapter resource is required")
        return adapter_payload(adapter_id, path, parse_query(query))
    if path.startswith("/api/runs/") and path.endswith("/report"):
        return run_report_payload(output_root, path)
    raise ValueError("not an API GET endpoint")


def post_payload(output_root: Path, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    if path.startswith("/api/agent/"):
        return post_agent_payload(output_root, path, payload)
    handler = _POST_HANDLERS.get(path)
    if handler is not None:
        return handler(output_root, payload)
    if path.startswith("/api/jobs/") and path.endswith("/cancel"):
        job_id = path.removeprefix("/api/jobs/").removesuffix("/cancel")
        return cancel_job_payload(output_root, unquote(job_id))
    raise ValueError("unknown endpoint")


def initialize_proxy_settings(output_root: Path) -> Path:
    path = configure_proxy_settings_path(proxy_settings_path(output_root))
    if not path.exists():
        save_adapter_proxy_settings(path, load_workbench_settings().proxy)
    return path


def run_report_payload(output_root: Path, path: str) -> dict[str, Any]:
    run_id = safe_run_id(path.removeprefix("/api/runs/").removesuffix("/report"))
    output_dir = output_root / run_id
    if not output_dir.exists():
        raise FileNotFoundError("run not found")
    config = read_json(output_dir / "config.json")
    dataset_path = Path(config.get("dataset", {}).get("path", ""))
    return build_workbench_report(output_dir, dataset_path=dataset_path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
