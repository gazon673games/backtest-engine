from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.accounting.futures import (
    FuturesExecutionCostSpec,
    FuturesInstrumentSpec,
    cost_breakdown,
)
from backtest_engine.adapters.models import DatasetBuildRequest, DatasetBuildResult
from backtest_engine.adapters.moex.pipeline.stages import repair_gaps
from backtest_engine.adapters.moex.sessions import MOEX_SESSION_WINDOWS
from backtest_engine.adapters.registry import get_adapter
from backtest_engine.data.session_gap_audit import session_gap_audit
from backtest_engine.workbench.api import async_run_payload
from backtest_engine.workbench.dataset_jobs import execute_dataset_build
from backtest_engine.workbench.datasets import (
    list_workbench_datasets,
    register_workbench_dataset,
)
from backtest_engine.workbench.progress import read_progress
from backtest_engine.workbench.worker import write_worker_failure


def test_session_gap_audit_ignores_clearing_and_overnight_closures() -> None:
    frame = pl.DataFrame(
        {
            "ts": [
                datetime(2024, 7, 1, 13, 55),
                datetime(2024, 7, 1, 14, 5),
                datetime(2024, 7, 2, 10, 0),
            ]
        }
    )

    audit = session_gap_audit(frame, "5m", MOEX_SESSION_WINDOWS)

    assert audit["gap_count"] == 0
    assert audit["expected_closure_count"] == 2
    assert audit["session_aware"] is True


def test_session_gap_audit_detects_missing_bar_inside_session() -> None:
    frame = pl.DataFrame({"ts": [datetime(2024, 7, 1, 10, 0), datetime(2024, 7, 1, 10, 15)]})

    audit = session_gap_audit(frame, "5m", MOEX_SESSION_WINDOWS)

    assert audit["gap_count"] == 1
    assert audit["gaps"][0]["session"] == "main_am"


def test_session_gap_audit_covers_moex_morning_session() -> None:
    frame = pl.DataFrame({"ts": [datetime(2024, 7, 1, 7, 0), datetime(2024, 7, 1, 7, 15)]})

    audit = session_gap_audit(frame, "5m", MOEX_SESSION_WINDOWS)

    assert audit["gap_count"] == 1
    assert audit["gaps"][0]["session"] == "main_am"


def test_gap_repair_never_synthesizes_overnight_bars() -> None:
    frame = pl.DataFrame(
        {
            "ts": [datetime(2024, 7, 1, 18, 50), datetime(2024, 7, 2, 10, 0)],
            "close": [10.0, 11.0],
        }
    )

    repaired, audit = repair_gaps(
        frame,
        timeframe="5m",
        policy="zero_volume_forward_fill",
        session_windows=MOEX_SESSION_WINDOWS,
    )

    assert repaired.height == 2
    assert audit["synthetic_bar_count"] == 0


def test_exchange_fee_from_instrument_spec_is_charged_per_contract() -> None:
    costs = cost_breakdown(
        contract_delta=2.0,
        turnover_notional_cash=1000.0,
        instrument=FuturesInstrumentSpec(exchange_fee_per_contract=3.5),
        execution=FuturesExecutionCostSpec(),
    )

    assert costs.commission_cost_cash == 7.0


def test_adapter_declares_ui_capabilities() -> None:
    capabilities = get_adapter("moex").info().capabilities

    assert capabilities.instrument_modes == ["nearest_futures", "security"]
    assert "offline" in capabilities.cache_policies
    assert capabilities.execution_defaults["slippage_ticks"] == 1.0
    assert capabilities.default_delay_minutes == 15
    assert capabilities.provider_supports_realtime is True
    assert capabilities.realtime_requires_subscription is True
    assert capabilities.adapter_realtime_enabled is False


def test_built_dataset_is_registered_in_workbench_catalog(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset" / "bars" / "nearest.parquet"
    reports = dataset_path.parent.parent / "reports"
    reports.mkdir(parents=True)
    dataset_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 2, 10, 0)],
            "exchange": ["MOEX"],
            "continuous_symbol": ["CR_NEAREST_5M"],
            "dataset_type": ["nearest_contract"],
            "adjustment_mode": ["unadjusted"],
        }
    ).write_parquet(dataset_path)
    artifact_names = [
        "approval.json",
        "pipeline_manifest.json",
        "raw.json",
        "instrument_registry.json",
        "instrument_registry_audit.json",
        "session_calendar.json",
        "session_calendar_audit.json",
    ]
    artifacts = {}
    for name in artifact_names:
        path = reports / name
        path.write_text("{}", encoding="utf-8")
        artifacts[name] = path
    request = DatasetBuildRequest(
        adapter_id="moex",
        engine="futures",
        market="forts",
        board="RFUD",
        security="CR",
        instrument_mode="nearest_futures",
        from_date="2024-01-01",
        till_date="2024-01-31",
    )
    result = DatasetBuildResult(
        adapter_id="moex",
        dataset_id="moex:fixture",
        dataset_path=dataset_path,
        output_root=dataset_path.parent.parent,
        approval_status="approved",
        approval_path=artifacts["approval.json"],
        manifest_path=artifacts["pipeline_manifest.json"],
        raw_snapshot_path=artifacts["raw.json"],
        timeframe="5m",
        row_count=1,
        instrument_registry_path=artifacts["instrument_registry.json"],
        instrument_registry_audit_path=artifacts["instrument_registry_audit.json"],
        calendar_path=artifacts["session_calendar.json"],
        calendar_audit_path=artifacts["session_calendar_audit.json"],
    )

    registered = register_workbench_dataset(tmp_path / "runs", request, result)
    catalog = list_workbench_datasets(tmp_path / "runs")

    assert registered.catalog_path is not None
    assert catalog["items"][0]["dataset_id"] == "moex:fixture"
    assert catalog["items"][0]["instrument_registry_path"].endswith("instrument_registry.json")


def test_dataset_build_worker_reports_catalogued_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = DatasetBuildResult(
        adapter_id="fixture",
        dataset_id="fixture:dataset",
        dataset_path=tmp_path / "nearest.parquet",
        output_root=tmp_path / "dataset",
        approval_status="approved",
        approval_path=tmp_path / "approval.json",
        manifest_path=tmp_path / "manifest.json",
        raw_snapshot_path=tmp_path / "raw.json",
        timeframe="5m",
        row_count=10,
    )

    class FakeAdapter:
        def build_dataset(
            self,
            _request: DatasetBuildRequest,
            observer: object | None = None,
        ) -> DatasetBuildResult:
            del observer
            return result

    monkeypatch.setattr(
        "backtest_engine.workbench.dataset_jobs.get_adapter",
        lambda _id: FakeAdapter(),
    )
    monkeypatch.setattr(
        "backtest_engine.workbench.dataset_jobs.register_workbench_dataset",
        lambda _root, _request, build_result: build_result,
    )
    output_root = tmp_path / "runs"
    execute_dataset_build(
        "dataset_job",
        {
            "run_output_root": str(output_root),
            "request": {
                "adapter_id": "fixture",
                "security": "DEMO",
                "from_date": "2024-01-01",
                "till_date": "2024-01-31",
            },
        },
    )

    progress = read_progress(output_root, "dataset_job")
    assert progress["status"] == "completed"
    assert progress["dataset_id"] == "fixture:dataset"


def test_dataset_worker_failure_is_visible_in_job_progress(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    request_path = tmp_path / "dataset_job.json"
    request_path.write_text(
        json.dumps(
            {
                "job_id": "failed_dataset",
                "job_type": "dataset",
                "payload": {"run_output_root": str(output_root)},
            }
        ),
        encoding="utf-8",
    )

    write_worker_failure(request_path, RuntimeError("ISS unavailable"))

    progress = read_progress(output_root, "failed_dataset")
    assert progress["status"] == "failed"
    assert progress["message"] == "ISS unavailable"


def test_async_workbench_job_runs_in_isolated_worker(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for JS strategy runtime tests")
    dataset_path = tmp_path / "bars.parquet"
    timestamps = [datetime(2024, 1, 2, 10, 0) + timedelta(minutes=5 * i) for i in range(3)]
    pl.DataFrame(
        {
            "source": ["fixture"] * 3,
            "exchange": ["MOEX"] * 3,
            "engine": ["futures"] * 3,
            "market": ["forts"] * 3,
            "board": ["RFUD"] * 3,
            "symbol": ["CRH4"] * 3,
            "continuous_symbol": ["CR_NEAREST_5M"] * 3,
            "timeframe": ["5m"] * 3,
            "ts": timestamps,
            "open": [10.0, 11.0, 12.0],
            "high": [10.1, 11.1, 12.1],
            "low": [9.9, 10.9, 11.9],
            "close": [10.0, 11.0, 12.0],
            "volume": [100.0] * 3,
            "value": [1000.0] * 3,
            "currency": ["RUB"] * 3,
            "active_contract": ["CRH4"] * 3,
            "source_contract_symbol": ["CRH4"] * 3,
            "dataset_type": ["nearest_contract"] * 3,
            "adjustment_mode": ["unadjusted"] * 3,
        }
    ).write_parquet(dataset_path)
    output_root = tmp_path / "runs"
    response = async_run_payload(
        output_root,
        {
            "dataset_path": str(dataset_path),
            "script_source": (
                'globalThis.STRATEGY_METADATA={name:"async_fixture",timeframe:"5m",'
                'schedule:"times",decision_times:["10:05"],'
                'missing_decision_policy:"close",params:[]};\n'
                'function onDecision(){return {target:1,contracts:1,reason:"fixture"};}\n'
            ),
            "from_date": "2024-01-02",
            "quality_approval_required": False,
            "allow_legacy_bar_timestamps": True,
            "run_id": "async_fixture",
        },
    )
    deadline = time.time() + 20
    progress = read_progress(output_root, response["run_id"])
    while progress["status"] not in {"completed", "failed"} and time.time() < deadline:
        time.sleep(0.1)
        progress = read_progress(output_root, response["run_id"])

    assert progress["status"] == "completed", progress
    assert (output_root / "async_fixture" / "metrics.json").exists()
