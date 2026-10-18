from __future__ import annotations

import json
import sys
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

from backtest_engine.workbench.jobs import WorkbenchJobManager
from backtest_engine.workbench.progress import read_progress


def test_duplicate_job_keeps_original_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WorkbenchJobManager(tmp_path / "runs", max_workers=1)
    monkeypatch.setattr(manager.executor, "submit", lambda *_args: Future())
    try:
        manager.submit("fixture", "run", {"source": "original"})
        record = manager.records["fixture"]
        with pytest.raises(ValueError, match="job already exists"):
            manager.submit("fixture", "run", {"source": "replacement"})
        saved = json.loads(record.request_path.read_text(encoding="utf-8"))
        assert saved["payload"] == {"source": "original"}
        assert manager.records["fixture"] is record
        assert read_progress(tmp_path / "runs", "fixture")["status"] == "queued"
    finally:
        manager.executor.shutdown(wait=True, cancel_futures=True)


def test_running_workbench_job_can_be_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backtest_engine.workbench.jobs.worker_command",
        lambda _path: [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    manager = WorkbenchJobManager(tmp_path / "runs", max_workers=1)
    try:
        manager.submit("cancel_fixture", "run", {})
        deadline = time.time() + 5
        while manager.records["cancel_fixture"].process is None and time.time() < deadline:
            time.sleep(0.02)
        progress = manager.cancel("cancel_fixture")
        manager.records["cancel_fixture"].future.result(timeout=10)

        assert progress["status"] == "cancelled"
        assert read_progress(tmp_path / "runs", "cancel_fixture")["status"] == "cancelled"
    finally:
        manager.executor.shutdown(wait=True, cancel_futures=True)
