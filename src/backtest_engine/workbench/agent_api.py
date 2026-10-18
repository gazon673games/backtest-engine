"""Agent-facing HTTP adapter over the same catalogs, queue and research services."""

from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from backtest_engine.workbench.agent_catalog import (
    build_request,
    datasets,
    strategies,
    strategy_contract,
)
from backtest_engine.workbench.agent_models import RunSpec
from backtest_engine.workbench.agent_results import artifact, result_root, results
from backtest_engine.workbench.api import async_run_payload
from backtest_engine.workbench.robustness_jobs import prepare_experiment, submit_experiment


def get_agent_payload(output_root: Path, path: str, query: str) -> dict[str, Any]:
    values = {k: v[0] for k, v in parse_qs(query).items()}
    offset, limit = int(values.get("offset", 0)), int(values.get("limit", 20))
    if path == "/api/agent/datasets":
        return datasets(
            output_root,
            values.get("query", ""),
            values.get("approved_only", "true") == "true",
            offset,
            limit,
        )
    if path == "/api/agent/strategies":
        return strategies(output_root, values.get("query", ""), offset, limit)
    kind, result_id = values.get("kind", "backtest"), values.get("result_id", "")
    if path == "/api/agent/results":
        return results(
            output_root,
            result_id,
            kind,
            offset,
            limit,
            values.get("group"),
            values.get("sort_by", "run_id"),
            values.get("descending", "false") == "true",
        )
    if path == "/api/agent/artifacts":
        columns = values["columns"].split(",") if values.get("columns") else None
        return artifact(
            output_root,
            result_id,
            kind,
            values["name"],
            offset,
            limit,
            columns,
            values.get("case_name"),
        )
    raise ValueError(f"unknown agent endpoint: {path}")


def post_agent_payload(output_root: Path, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    if path == "/api/agent/strategy-contract":
        return strategy_contract(output_root, payload["strategy_id"], payload.get("params"))
    if path == "/api/agent/backtests":
        request = build_request(output_root, RunSpec.model_validate(payload["run"]))
        raw = request.model_dump(mode="json")
        if payload.get("run_id"):
            run_id = payload["run_id"]
            if result_root(output_root, run_id, "backtest").exists():
                raise ValueError(f"native run already exists: {run_id}; choose a new run_id")
            raw["run_id"] = run_id
        return async_run_payload(output_root, raw)
    if path == "/api/agent/experiments/prepare":
        request = build_request(output_root, RunSpec.model_validate(payload["run"]))
        return prepare_experiment(
            output_root,
            {
                "base_request": request.model_dump(mode="json"),
                "protocol": payload["protocol"],
                "experiment_id": payload.get("experiment_id"),
            },
        )
    raise ValueError(f"unknown agent endpoint: {path}")


prepare_native_experiment = prepare_experiment
submit_native_experiment = submit_experiment
