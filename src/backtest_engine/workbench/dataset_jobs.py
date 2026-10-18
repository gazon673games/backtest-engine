from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backtest_engine.adapters.build_events import timeframe_minutes
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.registry import get_adapter
from backtest_engine.adapters.request_defaults import resolve_build_request
from backtest_engine.workbench.dataset_build_preview import WorkbenchDatasetBuildObserver
from backtest_engine.workbench.dataset_reuse import bind_reusable_dataset
from backtest_engine.workbench.datasets import register_workbench_dataset
from backtest_engine.workbench.jobs import submit_job
from backtest_engine.workbench.progress import write_progress
from backtest_engine.workbench.runner import safe_run_id


def submit_dataset_build(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("output_root", str(output_root.parent / "datasets"))
    request = DatasetBuildRequest.model_validate(payload)
    adapter = get_adapter(request.adapter_id)
    request = bind_reusable_dataset(output_root, resolve_build_request(request, adapter), adapter)
    preview_timeframe = str(payload.get("preview_timeframe") or request.timeframe)
    validate_preview_timeframe(request.timeframe, preview_timeframe)
    job_id = generated_dataset_job_id(request)
    job_payload = {
        "run_output_root": str(output_root),
        "request": request.model_dump(mode="json"),
        "preview_timeframe": preview_timeframe,
    }
    job = submit_job(output_root, job_id, "dataset", job_payload)
    return {"status": "accepted", "run_id": job_id, "job": job}


def execute_dataset_build(job_id: str, payload: dict[str, Any]) -> None:
    output_root = Path(payload["run_output_root"])
    request = DatasetBuildRequest.model_validate(payload["request"])
    observer = WorkbenchDatasetBuildObserver(
        output_root,
        job_id,
        str(payload.get("preview_timeframe") or request.timeframe),
    )
    observer.progress(
        stage="dataset_download",
        message=f"building {request.adapter_id}:{request.security}",
        pct=0.15,
    )
    result = get_adapter(request.adapter_id).build_dataset(request, observer=observer)
    observer.progress(
        stage="dataset_catalog",
        message="registering dataset and approvals",
        pct=0.9,
    )
    result = register_workbench_dataset(output_root, request, result)
    observer.complete(result.timeframe)
    write_progress(
        output_root,
        job_id,
        status="completed",
        stage="done",
        message="dataset completed",
        pct=1.0,
        extra={
            "job_type": "dataset",
            "dataset_id": result.dataset_id,
            "dataset_path": str(result.dataset_path),
            "approval_status": result.approval_status,
        },
    )


def generated_dataset_job_id(request: DatasetBuildRequest) -> str:
    encoded = json.dumps(request.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:8]
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return safe_run_id(f"dataset_{stamp}_{digest}")


def validate_preview_timeframe(source: str, preview: str) -> None:
    source_minutes = timeframe_minutes(source)
    preview_minutes = timeframe_minutes(preview)
    if preview_minutes < source_minutes or preview_minutes % source_minutes:
        raise ValueError(f"preview timeframe {preview} cannot be built from {source}")
