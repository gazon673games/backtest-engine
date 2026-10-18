from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def progress_root(output_root: Path) -> Path:
    return output_root.parent / "progress"


def write_progress(
    output_root: Path,
    run_id: str,
    *,
    status: str,
    stage: str,
    message: str,
    pct: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = progress_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "status": status,
        "stage": stage,
        "message": message,
        "pct": pct,
        "updated_at": datetime.now(UTC).isoformat(),
        **(extra or {}),
    }
    path = root / f"{run_id}.json"
    temporary = root / f".{run_id}.{os.getpid()}.{threading.get_ident()}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        replace_progress_file(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def replace_progress_file(temporary: Path, path: Path) -> None:
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            # Windows can briefly deny replacement while the UI is reading the old JSON.
            time.sleep(min(0.005 * (attempt + 1), 0.05))


def read_progress(output_root: Path, run_id: str) -> dict[str, Any]:
    path = progress_root(output_root) / f"{run_id}.json"
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {
                "run_id": run_id,
                "status": "unknown",
                "stage": "unknown",
                "message": "progress not found",
                "pct": 0.0,
            }
        except PermissionError:
            if attempt == 19:
                raise
            # Windows can briefly deny opening the JSON while a worker replaces it.
            time.sleep(min(0.005 * (attempt + 1), 0.05))
