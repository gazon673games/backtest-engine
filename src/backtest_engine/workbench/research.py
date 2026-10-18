"""Executes parameter sweeps and assembles reproducible robustness reports."""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.research.case_execution import execute_case_record
from backtest_engine.research.robustness.provenance import request_input_hashes, verify_hashes
from backtest_engine.research.validation.gate import (
    ResearchValidationConfig,
    generate_research_validation_report,
)
from backtest_engine.research.validation.io import write_table
from backtest_engine.workbench.jobs import submit_job
from backtest_engine.workbench.progress import write_progress
from backtest_engine.workbench.research_models import WorkbenchSweepRequest
from backtest_engine.workbench.runner import run_workbench_backtest, safe_run_id
from backtest_engine.workbench.sdk_catalog import freeze_run_request


def submit_sweep(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("base_request", {})
    payload["base_request"].setdefault("output_root", str(output_root))
    request = WorkbenchSweepRequest.model_validate(payload)
    request.base_request = freeze_run_request(request.base_request)
    request.input_hashes = request_input_hashes(request.base_request.model_dump(mode="json"))
    payload = request.model_dump(mode="json")
    sweep_id = safe_run_id(request.run_id or generated_sweep_id(request))
    payload["run_id"] = sweep_id
    request = WorkbenchSweepRequest.model_validate(payload)
    job = submit_job(output_root, sweep_id, "sweep", request.model_dump(mode="json"))
    return {
        "status": "accepted",
        "run_id": sweep_id,
        "run_count": request.run_count,
        "job": job,
    }


def execute_sweep(job_id: str, payload: dict[str, Any]) -> None:
    request = WorkbenchSweepRequest.model_validate(payload)
    base = freeze_run_request(request.base_request)
    if not request.input_hashes:
        request.input_hashes = request_input_hashes(base.model_dump(mode="json"))
    verify_hashes(request.input_hashes)
    experiment_path = research_root(base.output_root) / job_id
    experiment_path.mkdir(parents=True, exist_ok=True)
    rows = []
    fingerprint = hashlib.sha256(
        json.dumps(request.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()
    combinations = parameter_combinations(request.param_grid)
    for index, params in enumerate(combinations, start=1):
        write_progress(
            base.output_root,
            job_id,
            status="running",
            stage="parameter_sweep",
            message=f"run {index}/{len(combinations)}",
            pct=0.1 + 0.65 * index / len(combinations),
            extra={"job_type": "sweep", "completed_runs": index - 1},
        )
        run_id = safe_run_id(f"{job_id}_{index:03d}")
        run_request = base.model_copy(
            update={
                "run_id": run_id,
                "params": {**base.params, **{key: str(value) for key, value in params.items()}},
            }
        )

        def operation(run_request=run_request):
            verify_hashes(request.input_hashes)
            response = run_workbench_backtest(run_request)
            verify_hashes(request.input_hashes)
            return {"run_path": str(response.output_dir), "metrics": response.report["metrics"]}

        record = execute_case_record(
            name=run_id,
            fingerprint=fingerprint,
            checkpoint=experiment_path / "checkpoints" / f"{run_id}.json",
            operation=operation,
        )
        if record["status"] == "completed":
            rows.append(
                {
                    "status": "completed",
                    **summary_row(run_id, Path(record["run_path"]), params, record["metrics"]),
                }
            )
        else:
            rows.append(
                {
                    "run_id": run_id,
                    "run_path": "",
                    "status": "failed",
                    "error": record["error"],
                    **params,
                }
            )
    write_table(experiment_path / "summary", pl.DataFrame(rows, orient="row"))
    verify_hashes(request.input_hashes)
    write_progress(
        base.output_root,
        job_id,
        status="running",
        stage="research_validation",
        message="building walk-forward and robustness reports",
        pct=0.82,
        extra={"job_type": "sweep", "completed_runs": len(rows)},
    )
    failures = [row for row in rows if row["status"] == "failed"]
    if failures:
        acceptance_status = "rejected"
        warnings = [f"{row['run_id']}: {row['error']}" for row in failures]
    else:
        validation = generate_research_validation_report(
            ResearchValidationConfig(
                experiment_path=experiment_path,
                output_path=experiment_path / "validation",
                split_date=request.split_date,
                group_columns=list(request.param_grid),
                bootstrap_iterations=request.bootstrap_iterations,
                cost_stress_bps=request.cost_stress_bps,
                max_parameter_runs_without_warning=request.max_runs,
            )
        )
        acceptance_status = validation.acceptance_status
        warnings = validation.warnings
    result = {
        "status": "completed",
        "run_id": job_id,
        "run_count": len(rows),
        "experiment_path": str(experiment_path),
        "validation_path": str(experiment_path / "validation"),
        "acceptance_status": acceptance_status,
        "warning_count": len(warnings),
        "warnings": warnings,
        "failed_run_count": len(failures),
    }
    write_json(experiment_path / "result.json", result)
    write_progress(
        base.output_root,
        job_id,
        status="completed",
        stage="done",
        message="research sweep completed",
        pct=1.0,
        extra={"job_type": "sweep", **result},
    )


def research_report(output_root: Path, sweep_id: str) -> dict[str, Any]:
    path = research_root(output_root) / safe_run_id(sweep_id)
    if not path.exists():
        raise FileNotFoundError(f"research sweep not found: {sweep_id}")
    validation = path / "validation"
    return {
        **read_json(path / "result.json"),
        "summary": read_records(path / "summary.json"),
        "walk_forward": read_records(validation / "walk_forward.json"),
        "rolling_walk_forward": read_records(validation / "rolling_walk_forward.json"),
        "locked_holdout": read_records(validation / "locked_holdout.json"),
        "robustness": read_records(validation / "robustness.json"),
        "guardrails": read_records(validation / "guardrails.json"),
        "bootstrap": read_records(validation / "bootstrap.json"),
        "cost_stress": read_records(validation / "cost_stress.json"),
        "regime": read_records(validation / "regime.json"),
        "gate": read_json(validation / "research_gate.json"),
    }


def parameter_combinations(
    grid: dict[str, list[str | int | float | bool]],
) -> list[dict[str, Any]]:
    names = list(grid)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(grid[name] for name in names))
    ]


def summary_row(
    run_id: str,
    output_dir: Path | None,
    params: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_path": str(output_dir or ""),
        **params,
        **{key: value for key, value in metrics.items() if isinstance(value, int | float | str)},
    }


def generated_sweep_id(request: WorkbenchSweepRequest) -> str:
    digest = hashlib.sha256(
        json.dumps(request.param_grid, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"sweep_{stamp}_{digest}"


def research_root(output_root: Path) -> Path:
    return output_root.parent / "research"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
