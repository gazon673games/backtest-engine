"""Prepare immutable research plans and execute them in the shared Workbench queue."""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from backtest_engine.research.robustness.models import Protocol
from backtest_engine.research.robustness.planning import request_payload
from backtest_engine.research.robustness.provenance import read_json, write_json
from backtest_engine.research.robustness.runner import execute_protocol, freeze_protocol
from backtest_engine.workbench.jobs import submit_job
from backtest_engine.workbench.models import WorkbenchRunRequest
from backtest_engine.workbench.progress import read_progress, write_progress


def experiment_root(output_root: Path, experiment_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", experiment_id):
        raise ValueError("experiment_id must contain 1..80 letters, digits, '_' or '-'")
    return output_root.resolve().parent / "research" / "protocols" / experiment_id


def prepare_experiment(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    experiment_id = payload.get("experiment_id") or (
        "robustness_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    )
    root = experiment_root(output_root, experiment_id)
    if root.exists():
        raise ValueError(f"experiment already exists: {experiment_id}; use submit with resume")
    raw_protocol = dict(payload["protocol"])
    if {"output_root", "base_request"} & raw_protocol.keys():
        raise ValueError("protocol output_root/base_request are allocated by Workbench")
    request = WorkbenchRunRequest.model_validate(
        request_payload(payload["base_request"], Path.cwd())
    )
    protocol = Protocol.model_validate(
        {**raw_protocol, "output_root": root, "base_request": root / "base_request.json"}
    )
    root.mkdir(parents=True)
    write_json(root / "base_request.json", request.model_dump(mode="json"))
    write_json(root / "protocol.json", protocol.model_dump(mode="json"))
    manifest = freeze_protocol(root / "protocol.json")
    return {
        "status": "prepared",
        "experiment_id": experiment_id,
        "case_count": len(manifest["plan"]["cases"]) + bool(protocol.fresh_selected_group),
        "fingerprint": manifest["fingerprint"],
        "experiment_path": str(root),
    }


def submit_experiment(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(payload["experiment_id"])
    root = experiment_root(output_root, experiment_id)
    manifest = read_json(root / "manifest.json")
    resume = payload.get("resume", False)
    if not isinstance(resume, bool):
        raise ValueError("resume must be a boolean")
    current = read_progress(output_root, experiment_id)
    if current.get("status") in {"queued", "running"}:
        return {"status": "accepted", "experiment_id": experiment_id, "job": current}
    if current.get("status") in {"completed", "failed", "cancelled"} and not resume:
        raise ValueError("experiment was already submitted; use resume=true or prepare a new plan")
    job = submit_job(
        output_root,
        experiment_id,
        "robustness",
        {
            "output_root": str(output_root.resolve()),
            "experiment_id": experiment_id,
            "manifest": manifest,
            "resume": resume,
        },
    )
    return {"status": "accepted", "experiment_id": experiment_id, "job": job}


def execute_experiment(job_id: str, payload: dict[str, Any]) -> None:
    output_root = Path(payload["output_root"])
    root = experiment_root(output_root, job_id)
    manifest = read_json(root / "manifest.json")
    if manifest != payload["manifest"]:
        raise ValueError(f"prepared manifest changed for experiment {job_id}")

    def progress(completed: int, total: int, name: str, status: str):
        write_progress(
            output_root,
            job_id,
            status="running",
            stage="robustness",
            message=f"{name}: {status}",
            pct=0.1 + 0.8 * completed / total,
            extra={
                "job_type": "robustness",
                "completed_runs": completed,
                "planned_runs": total,
                "experiment_path": str(root),
            },
        )

    planned = len(manifest["plan"]["cases"]) + bool(
        manifest["plan"]["protocol"].get("fresh_selected_group")
    )
    progress(0, planned, "protocol", "starting")
    result = execute_protocol(manifest, resume=payload["resume"], progress_callback=progress)
    write_progress(
        output_root,
        job_id,
        status=result["status"],
        stage="done",
        pct=1,
        message=f"robustness: {result['case_count']} cases, {result['failed_count']} failures",
        extra={"job_type": "robustness", "experiment_path": str(root), **result},
    )


def experiment_status(output_root: Path, experiment_id: str) -> dict[str, Any]:
    root = experiment_root(output_root, experiment_id)
    manifest = read_json(root / "manifest.json")
    result = read_json(root / "result.json") if (root / "result.json").exists() else {}
    return {
        "experiment_id": experiment_id,
        "fingerprint": manifest["fingerprint"],
        "job": read_progress(output_root, experiment_id),
        **result,
    }
