"""Thin CLI adapter for preparing and submitting research to the common HTTP queue."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from backtest_engine.research.robustness.planning import load_protocol, request_payload
from backtest_engine.research.robustness.provenance import read_json
from backtest_engine.workbench.http_client import WorkbenchClient
from backtest_engine.workbench.settings import load_workbench_settings


def client(url: str | None) -> WorkbenchClient:
    port = load_workbench_settings().server.port
    return WorkbenchClient(url or os.environ.get("WORKBENCH_URL") or f"http://127.0.0.1:{port}")


def register(app: typer.Typer):
    @app.command("robustness-prepare")
    def prepare(
        config: Annotated[Path, typer.Option(exists=True)],
        experiment_id: str | None = None,
        url: str | None = None,
    ):
        """Freeze a JSON protocol on the running Workbench, without executing trades."""
        protocol, _cases = load_protocol(config)
        if protocol.base_request is None:
            raise typer.BadParameter("queued protocols require base_request")
        base = request_payload(read_json(protocol.base_request), protocol.base_request.parent)
        raw = protocol.model_dump(mode="json", exclude={"output_root", "base_request"})
        for case in raw["cases"]:
            case["request_overrides"] = request_payload(
                case["request_overrides"], protocol.base_request.parent
            )
        result = client(url).call(
            "/api/research/protocols/prepare",
            payload={
                "base_request": base,
                "protocol": raw,
                "experiment_id": experiment_id,
            },
        )
        typer.echo(json.dumps(result, indent=2))

    @app.command("robustness-submit")
    def submit(experiment_id: str, resume: bool = False, url: str | None = None):
        """Submit a prepared experiment to the same queue used by desktop and MCP."""
        result = client(url).call(
            "/api/research/protocols/submit",
            payload={
                "experiment_id": experiment_id,
                "resume": resume,
            },
        )
        typer.echo(json.dumps(result, indent=2))

    @app.command("robustness-status")
    def status(experiment_id: str, url: str | None = None):
        """Read progress and a compact paged result table."""
        result = client(url).call(
            "/api/agent/results",
            query={
                "result_id": experiment_id,
                "kind": "robustness",
            },
        )
        typer.echo(json.dumps(result, indent=2))
