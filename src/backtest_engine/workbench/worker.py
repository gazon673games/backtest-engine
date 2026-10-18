from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from backtest_engine.workbench.dataset_jobs import execute_dataset_build
from backtest_engine.workbench.history import (
    append_workbench_history,
    failure_record,
    history_path_for,
)
from backtest_engine.workbench.live_progress import RunProgressReporter
from backtest_engine.workbench.models import WorkbenchRunRequest
from backtest_engine.workbench.portfolio_jobs import execute_portfolio
from backtest_engine.workbench.progress import write_progress
from backtest_engine.workbench.research import execute_sweep
from backtest_engine.workbench.robustness_jobs import execute_experiment
from backtest_engine.workbench.runner import run_workbench_backtest


def execute_job(request_path: Path) -> None:
    job = json.loads(request_path.read_text(encoding="utf-8"))
    job_id = str(job["job_id"])
    job_type = str(job["job_type"])
    payload = job.get("payload", {})
    handlers = {
        "run": execute_run,
        "portfolio": execute_portfolio,
        "sweep": execute_sweep,
        "dataset": execute_dataset_build,
        "robustness": execute_experiment,
    }
    handler = handlers.get(job_type)
    if handler is None:
        raise ValueError(f"unknown workbench job type: {job_type}")
    handler(job_id, payload)


def execute_run(job_id: str, payload: dict[str, Any]) -> None:
    request = WorkbenchRunRequest.model_validate(payload)
    write_progress(
        request.output_root,
        job_id,
        status="running",
        stage="backtest",
        message="running backtest",
        pct=0.03,
        extra={"job_type": "run"},
    )
    try:
        response = run_workbench_backtest(
            request,
            progress_callback=RunProgressReporter(request.output_root, job_id),
        )
    except Exception as exc:
        append_workbench_history(
            history_path_for(request.output_root),
            failure_record(request, run_id=job_id, error=str(exc)),
        )
        raise
    write_progress(
        request.output_root,
        job_id,
        status="completed",
        stage="done",
        message="run completed",
        pct=1.0,
        extra={
            "job_type": "run",
            "output_dir": str(response.output_dir),
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: python -m backtest_engine.workbench.worker REQUEST.json")
    request_path = Path(args[0])
    try:
        execute_job(request_path)
    except Exception as exc:
        write_worker_failure(request_path, exc)
        return 1
    return 0


def write_worker_failure(request_path: Path, exc: Exception) -> None:
    try:
        job = json.loads(request_path.read_text(encoding="utf-8"))
        payload = job.get("payload", {})
        raw_output_root = (
            payload.get("output_root")
            or payload.get("run_output_root")
            or payload.get("base_request", {}).get("output_root")
        )
        output_root = Path(raw_output_root)
        write_progress(
            output_root,
            str(job["job_id"]),
            status="failed",
            stage="failed",
            message=str(exc),
            pct=1.0,
            extra={"job_type": str(job.get("job_type", "unknown"))},
        )
    except Exception:
        return


if __name__ == "__main__":
    raise SystemExit(main())
