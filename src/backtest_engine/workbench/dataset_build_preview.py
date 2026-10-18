"""Stores immutable provisional chart chunks produced by dataset workers."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from backtest_engine.workbench.progress import write_progress

PREVIEW_CHUNK_ROWS = 100
PREVIEW_READ_CHUNKS = 4


class DatasetBuildPreviewStore:
    def __init__(self, output_root: Path, job_id: str) -> None:
        self.output_root = output_root
        self.job_id = job_id
        self.root = preview_root(output_root) / job_id
        self.chunks = self.root / "chunks"
        self.sequence = 0
        self.pending: list[dict[str, Any]] = []

    def reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.chunks.mkdir(parents=True, exist_ok=True)
        self.pending = []
        self._write_manifest(status="running", timeframe=None)

    def append(self, rows: list[dict[str, Any]], timeframe: str) -> None:
        self.pending.extend(rows)
        if self.sequence == 0 and self.pending:
            self._flush(len(self.pending), timeframe)
        while len(self.pending) >= PREVIEW_CHUNK_ROWS:
            self._flush(PREVIEW_CHUNK_ROWS, timeframe)
        self._write_manifest(status="running", timeframe=timeframe)

    def complete(self, timeframe: str) -> None:
        if self.pending:
            self._flush(len(self.pending), timeframe)
        self._write_manifest(status="completed", timeframe=timeframe)

    def _flush(self, count: int, timeframe: str) -> None:
        chunk = self.pending[:count]
        self.pending = self.pending[count:]
        self.sequence += 1
        atomic_json(
            self.chunks / f"{self.sequence:08d}.json",
            {"sequence": self.sequence, "timeframe": timeframe, "rows": chunk},
        )

    def _write_manifest(self, *, status: str, timeframe: str | None) -> None:
        atomic_json(
            self.root / "manifest.json",
            {
                "job_id": self.job_id,
                "status": status,
                "latest_sequence": self.sequence,
                "timeframe": timeframe,
                "provisional": status != "completed",
            },
        )


class WorkbenchDatasetBuildObserver:
    def __init__(self, output_root: Path, job_id: str, preview_timeframe: str) -> None:
        self.output_root = output_root
        self.job_id = job_id
        self.preview_timeframe = preview_timeframe
        self.store = DatasetBuildPreviewStore(output_root, job_id)
        self.store.reset()

    def progress(self, *, stage: str, message: str, pct: float) -> None:
        write_progress(
            self.output_root,
            self.job_id,
            status="running",
            stage=stage,
            message=message,
            pct=pct,
            extra={"job_type": "dataset"},
        )

    def preview(self, rows: list[dict[str, Any]], *, timeframe: str) -> None:
        if rows:
            self.store.append(rows, timeframe)

    def complete(self, timeframe: str) -> None:
        self.store.complete(timeframe)


def read_dataset_build_preview(
    output_root: Path,
    job_id: str,
    *,
    after: int,
) -> dict[str, Any]:
    root = preview_root(output_root) / job_id
    manifest = read_json(root / "manifest.json")
    latest = int(manifest.get("latest_sequence", 0))
    rows: list[dict[str, Any]] = []
    next_sequence = after
    upper = min(latest, after + PREVIEW_READ_CHUNKS)
    for sequence in range(after + 1, upper + 1):
        payload = read_json(root / "chunks" / f"{sequence:08d}.json")
        rows.extend(payload.get("rows", []))
        next_sequence = sequence
    return {
        "job_id": job_id,
        "rows": rows,
        "next_sequence": next_sequence,
        "latest_sequence": latest,
        "timeframe": manifest.get("timeframe"),
        "status": manifest.get("status", "waiting"),
        "provisional": bool(manifest.get("provisional", True)),
    }


def preview_root(output_root: Path) -> Path:
    return output_root.parent / "dataset-previews"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
