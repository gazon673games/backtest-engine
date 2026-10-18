"""Offline regressions for queue snapshots, process state and HTTP boundaries."""

import json
import multiprocessing as mp
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from backtest_engine.data.catalog import DataCatalogEntry, load_data_catalog
from backtest_engine.strategies.runtime import StrategyMetadata
from backtest_engine.workbench import server
from backtest_engine.workbench.datasets import upsert_catalog_entry
from backtest_engine.workbench.jobs import WorkbenchJobManager
from backtest_engine.workbench.models import WorkbenchRunRequest
from backtest_engine.workbench.progress import read_progress, write_progress
from backtest_engine.workbench.runner import build_backtest_config
from backtest_engine.workbench.sdk_catalog import freeze_run_request


def register_entries(path, barrier, prefix):
    barrier.wait(timeout=15)
    for index in range(10):
        upsert_catalog_entry(
            Path(path),
            DataCatalogEntry(
                dataset_id=f"{prefix}-{index}", dataset_path=Path(path).parent / "bars.parquet"
            ),
        )


def test_catalog_keeps_every_worker_registration(tmp_path):
    path = tmp_path / "catalog.json"
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    workers = [
        ctx.Process(target=register_entries, args=(str(path), barrier, name))
        for name in ("first", "second")
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
        assert len(load_data_catalog(path).entries) == 20
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)


def test_http_rejects_foreign_origin_and_static_sibling(tmp_path, monkeypatch):
    static = tmp_path / "web"
    static.mkdir()
    sibling = tmp_path / "web-private"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("private", encoding="utf-8")
    calls = []
    monkeypatch.setattr(server, "STATIC_ROOT", static)
    monkeypatch.setattr(server, "post_payload", lambda *args: calls.append(args) or {"ok": True})
    http = ThreadingHTTPServer(("127.0.0.1", 0), server.handler_class(tmp_path / "runs"))
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", http.server_port, timeout=5)
    try:
        connection.request(
            "POST",
            "/api/strategy-metadata",
            "{}",
            headers={"Origin": "https://foreign.example", "Content-Type": "text/plain"},
        )
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        assert calls == []
        connection.request(
            "POST", "/api/strategy-metadata", "{}", headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        assert len(calls) == 1
        connection.request(
            "POST", "/api/sdk-source/file", "{}", headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        connection.request("GET", "/../web-private/secret.txt")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
    finally:
        connection.close()
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


def test_worker_spawn_failure_is_terminal_and_manager_closes(tmp_path, monkeypatch):
    def fail_spawn(*args, **kwargs):
        raise OSError("spawn rejected")

    monkeypatch.setattr("backtest_engine.workbench.jobs.subprocess.Popen", fail_spawn)
    manager = WorkbenchJobManager(tmp_path / "runs", 1)
    try:
        manager.submit("failed", "backtest", {})
        manager.records["failed"].future.result(timeout=5)
        assert read_progress(manager.output_root, "failed")["status"] == "failed"
        assert manager.cancel("failed")["status"] == "failed"
        manager.submit("another", "backtest", {})
        manager.records["another"].future.result(timeout=5)
        assert "failed" not in manager.records
    finally:
        manager.close()
    assert manager.records == {}


def test_resubmit_waits_for_completed_worker_to_exit(tmp_path, monkeypatch):
    published, release = Event(), Event()
    manager = WorkbenchJobManager(tmp_path / "runs", 2)
    attempts = []

    def spawn(*args, **kwargs):
        attempts.append(len(attempts))
        first = len(attempts) == 1

        def wait():
            write_progress(
                manager.output_root,
                "retry",
                status="completed",
                stage="done",
                message="done",
                pct=1,
            )
            if first:
                published.set()
                assert release.wait(timeout=5)
            return 0

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr("backtest_engine.workbench.jobs.subprocess.Popen", spawn)
    retry = Thread(target=manager.submit, args=("retry", "backtest", {}))
    try:
        manager.submit("retry", "backtest", {})
        assert published.wait(timeout=5)
        retry.start()
        retry.join(timeout=0.1)
        assert retry.is_alive()
        assert read_progress(manager.output_root, "retry")["status"] == "completed"
        release.set()
        retry.join(timeout=5)
        assert not retry.is_alive()
        manager.records["retry"].future.result(timeout=5)
        assert read_progress(manager.output_root, "retry")["status"] == "completed"
        assert len(attempts) == 2
    finally:
        release.set()
        if retry.ident is not None:
            retry.join(timeout=5)
        manager.close()


@pytest.mark.parametrize("extension,language", [("py", "python"), ("js", "javascript")])
def test_plain_source_is_frozen_before_worker_launch(tmp_path, extension, language):
    script = tmp_path / f"strategy.{extension}"
    source = (
        "def generate_signal(ctx): return 0\n"
        if extension == "py"
        else "function generateSignal(ctx) { return 0; }\n"
    )
    script.write_text(source, encoding="utf-8")
    frozen = freeze_run_request(
        WorkbenchRunRequest(dataset_path=tmp_path / "bars.parquet", script_path=script)
    )
    script.write_text("changed after submit", encoding="utf-8")
    assert frozen.script_path is None
    assert frozen.script_source == source
    assert frozen.script_language == language


@pytest.mark.parametrize("mode", ["fixed_contracts", "notional"])
def test_sizing_request_survives_effective_config(tmp_path, mode):
    from test_intraday_event_workflow_support import write_intraday_dataset

    write_intraday_dataset(tmp_path).rename(tmp_path / "bars.parquet")
    request = WorkbenchRunRequest(
        dataset_path=tmp_path / "bars.parquet",
        script_source="x",
        sizing_mode=mode,
        fixed_contracts=7,
        target_notional=5000,
    )
    config = build_backtest_config(
        request, "sizing", tmp_path / "strategy.py", StrategyMetadata(name="audit", timeframe="5m")
    )
    assert config.portfolio.sizing_mode == mode
    assert config.portfolio.fixed_contracts == 7
    assert config.portfolio.target_notional == 5000


def test_sweep_records_failure_and_continues(tmp_path, monkeypatch):
    from backtest_engine.workbench import research

    calls = []
    (tmp_path / "bars.parquet").write_bytes(b"frozen fixture")

    def run(request):
        calls.append(request.params["case"])
        if request.params["case"] == "2":
            raise ValueError("case failed")
        return SimpleNamespace(
            output_dir=tmp_path / request.run_id, report={"metrics": {"total_return": 0.1}}
        )

    monkeypatch.setattr(research, "run_workbench_backtest", run)
    payload = {
        "base_request": {
            "dataset_path": str(tmp_path / "bars.parquet"),
            "script_source": "function generateSignal(ctx) { return 0; }",
            "output_root": str(tmp_path / "runs"),
        },
        "param_grid": {"case": [1, 2, 3]},
        "split_date": "2024-07-01",
    }
    research.execute_sweep("audit", payload)
    directory = research.research_root(tmp_path / "runs") / "audit"
    summary = json.loads((directory / "summary.json").read_text())
    assert calls == ["1", "2", "3"]
    assert [row["status"] for row in summary] == ["completed", "failed", "completed"]
    assert len(list((directory / "checkpoints").glob("*.json"))) == 3
