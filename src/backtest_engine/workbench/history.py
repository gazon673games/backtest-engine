from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backtest_engine.workbench.models import WorkbenchRunRequest


def history_path_for(output_root: Path) -> Path:
    return output_root.parent / "history.json"


def load_workbench_history(path: Path) -> dict[str, Any]:
    runs = legacy_runs(path)
    root = history_records_root(path)
    if root.exists():
        for record_path in root.glob("*.json"):
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(record, dict):
                runs[str(record.get("run_id", record_path.stem))] = record
    ordered = sorted(runs.values(), key=lambda item: str(item.get("created_at", "")), reverse=True)
    return {"version": 2, "runs": ordered[:200]}


def append_workbench_history(path: Path, record: dict[str, Any]) -> None:
    root = history_records_root(path)
    root.mkdir(parents=True, exist_ok=True)
    run_id = str(record.get("run_id") or "unknown")
    temporary = root / f".{run_id}.tmp"
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(root / f"{run_id}.json")


def legacy_runs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("runs", []) if isinstance(payload, dict) else []
    return {str(row.get("run_id", index)): row for index, row in enumerate(rows)}


def history_records_root(path: Path) -> Path:
    return path.with_name("history_records")


def success_record(
    request: WorkbenchRunRequest,
    *,
    run_id: str,
    output_dir: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    return base_record(request, run_id, "completed") | {
        "output_dir": str(output_dir),
        "metrics": report.get("metrics", {}),
    }


def failure_record(
    request: WorkbenchRunRequest,
    *,
    run_id: str,
    error: str,
) -> dict[str, Any]:
    return base_record(request, run_id, "failed") | {"error": error}


def base_record(request: WorkbenchRunRequest, run_id: str, status: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": status,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(request.dataset_path),
        "script_path": None if request.script_path is None else str(request.script_path),
        "params": dict(request.params),
        "from_date": request.from_date,
        "decision_times": list(request.decision_times),
    }
