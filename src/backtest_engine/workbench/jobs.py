"""Queues cancellable jobs and executes each one in an isolated process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backtest_engine.workbench.progress import progress_root, read_progress, write_progress
from backtest_engine.workbench.settings import load_workbench_settings


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    request_path: Path
    future: Future[None] | None = None
    process: subprocess.Popen[bytes] | None = None
    cancel_requested: bool = False


class WorkbenchJobManager:
    def __init__(self, output_root: Path, max_workers: int) -> None:
        self.output_root = output_root
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="workbench-job",
        )
        self.records: dict[str, JobRecord] = {}
        self.lock = threading.Lock()

    def submit(self, job_id: str, job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            previous = self.records.get(job_id)
        # The worker can publish a result before its process exits. Finish its
        # supervision before a new attempt can reuse the same progress file.
        if previous is not None and previous.future is not None:
            if not previous.future.done() and job_finished(self.output_root, job_id):
                previous.future.result()
        with self.lock:
            self.records = {
                key: value
                for key, value in self.records.items()
                if value.future is None or not value.future.done()
            }
            if job_id in self.records:
                raise ValueError(f"job already exists: {job_id}")
            request_path = write_job_request(self.output_root, job_id, job_type, payload)
            record = JobRecord(job_id=job_id, job_type=job_type, request_path=request_path)
            self.records[job_id] = record
            write_progress(
                self.output_root,
                job_id,
                status="queued",
                stage="queued",
                message="waiting for worker slot",
                pct=0.0,
                extra={"job_type": job_type},
            )
            record.future = self.executor.submit(self._execute, record)
        return read_progress(self.output_root, job_id)

    def close(self) -> None:
        with self.lock:
            active = [
                record.job_id
                for record in self.records.values()
                if record.future is not None and not record.future.done()
            ]
        for job_id in active:
            self.cancel(job_id)
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.records.clear()

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            record = self.records.get(job_id)
            current = read_progress(self.output_root, job_id)
            if current.get("status") in {"completed", "failed", "cancelled"}:
                return current
            if record is None:
                raise FileNotFoundError(f"active job not found: {job_id}")
            record.cancel_requested = True
            future = record.future
            process = record.process
        if future is not None and future.cancel():
            return write_cancelled(self.output_root, record)
        if process is not None and process.poll() is None:
            try:
                terminate_process_tree(process)
            except OSError as exc:
                return write_progress(
                    self.output_root,
                    job_id,
                    status="cancelling",
                    stage="cancel_failed",
                    message=f"worker termination failed: {exc}",
                    pct=0.0,
                )
            return write_cancelled(self.output_root, record)
        return write_progress(
            self.output_root,
            job_id,
            status="cancelling",
            stage="cancel_requested",
            message="waiting for worker termination",
            pct=0.0,
        )

    def _execute(self, record: JobRecord) -> None:
        try:
            self._execute_worker(record)
        except Exception as exc:
            write_progress(
                self.output_root,
                record.job_id,
                status="failed",
                stage="worker_start",
                message=f"worker failed: {exc}",
                pct=1.0,
                extra={"job_type": record.job_type},
            )
        finally:
            with self.lock:
                record.process = None

    def _execute_worker(self, record: JobRecord) -> None:
        if record.cancel_requested:
            write_cancelled(self.output_root, record)
            return
        log_path = job_root(self.output_root) / f"{record.job_id}.log"
        write_progress(
            self.output_root,
            record.job_id,
            status="running",
            stage="worker_start",
            message="starting isolated worker",
            pct=0.05,
            extra={"job_type": record.job_type, "log_path": str(log_path)},
        )
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                worker_command(record.request_path),
                cwd=Path.cwd(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            with self.lock:
                record.process = process
                should_cancel = record.cancel_requested
            if should_cancel and process.poll() is None:
                terminate_process_tree(process)
            return_code = process.wait()
        with self.lock:
            record.process = None
        if record.cancel_requested:
            write_cancelled(self.output_root, record)
        elif return_code != 0 or not job_finished(self.output_root, record.job_id):
            current = read_progress(self.output_root, record.job_id)
            if current.get("status") != "failed":
                write_progress(
                    self.output_root,
                    record.job_id,
                    status="failed",
                    stage="worker_exit",
                    message=f"worker exited with code {return_code}",
                    pct=1.0,
                    extra={"job_type": record.job_type, "log_path": str(log_path)},
                )


def manager_for(output_root: Path) -> WorkbenchJobManager:
    key = str(output_root.resolve())
    with MANAGERS_LOCK:
        manager = MANAGERS.get(key)
        if manager is None:
            workers = configured_max_concurrency()
            manager = WorkbenchJobManager(output_root, workers)
            MANAGERS[key] = manager
        return manager


def submit_job(
    output_root: Path,
    job_id: str,
    job_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return manager_for(output_root).submit(job_id, job_type, payload)


def cancel_job(output_root: Path, job_id: str) -> dict[str, Any]:
    return manager_for(output_root).cancel(job_id)


def list_jobs(output_root: Path) -> dict[str, Any]:
    root = progress_root(output_root)
    items = []
    if root.exists():
        for path in root.glob("*.json"):
            try:
                items.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    items.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return {
        "max_concurrency": configured_max_concurrency(),
        "items": items[:200],
    }


def configured_max_concurrency() -> int:
    override = os.environ.get("WORKBENCH_MAX_JOBS")
    if override is not None:
        return max(1, int(override))
    return load_workbench_settings().jobs.max_concurrency


def write_job_request(
    output_root: Path,
    job_id: str,
    job_type: str,
    payload: dict[str, Any],
) -> Path:
    root = job_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{job_id}.json"
    record = {"job_id": job_id, "job_type": job_type, "payload": payload}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def job_root(output_root: Path) -> Path:
    return output_root.parent / "jobs"


def job_finished(output_root: Path, job_id: str) -> bool:
    return read_progress(output_root, job_id).get("status") in {
        "completed",
        "failed",
        "cancelled",
    }


def write_cancelled(output_root: Path, record: JobRecord) -> dict[str, Any]:
    return write_progress(
        output_root,
        record.job_id,
        status="cancelled",
        stage="cancelled",
        message="job cancelled",
        pct=1.0,
        extra={"job_type": record.job_type},
    )


def worker_command(request_path: Path) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--worker", str(request_path)]
    return [sys.executable, "-m", "backtest_engine.workbench.worker", str(request_path)]


def terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/pid", str(process.pid), "/t", "/f"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode and process.poll() is None:
            process.kill()
    else:
        process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired as exc:
        raise OSError(f"Worker {process.pid} did not stop after termination") from exc


MANAGERS: dict[str, WorkbenchJobManager] = {}
MANAGERS_LOCK = threading.Lock()


def close_manager(output_root: Path) -> None:
    with MANAGERS_LOCK:
        manager = MANAGERS.pop(str(output_root.resolve()), None)
    if manager is not None:
        manager.close()
