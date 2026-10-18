from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.workbench.research import execute_sweep, research_report


def test_workbench_parameter_sweep_builds_walk_forward_report(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for JS strategy runtime tests")
    dataset_path = write_research_dataset(tmp_path)
    output_root = tmp_path / "runs"
    payload = {
        "base_request": {
            "dataset_path": str(dataset_path),
            "script_source": strategy_source(),
            "from_date": "2024-01-02",
            "end_of_data_position_policy": "mark",
            "quality_approval_required": False,
            "allow_legacy_bar_timestamps": True,
            "output_root": str(output_root),
            "initial_capital": 1000.0,
        },
        "param_grid": {"bias": [-1, 1]},
        "split_date": "2024-01-05",
        "run_id": "sweep_fixture",
    }

    execute_sweep("sweep_fixture", payload)
    report = research_report(output_root, "sweep_fixture")

    assert report["status"] == "completed"
    assert len(report["summary"]) == 2
    assert len(report["walk_forward"]) == 2
    assert report["gate"]["acceptance_status"] in {"accepted", "rejected"}


def write_research_dataset(tmp_path: Path) -> Path:
    timestamps = []
    closes = []
    for day_index in range(6):
        start = datetime(2024, 1, 2, 10, 0) + timedelta(days=day_index)
        timestamps.extend([start, start + timedelta(minutes=5)])
        closes.extend([10.0 + day_index, 10.5 + day_index])
    count = len(timestamps)
    path = tmp_path / "research_bars.parquet"
    pl.DataFrame(
        {
            "source": ["fixture"] * count,
            "exchange": ["MOEX"] * count,
            "engine": ["futures"] * count,
            "market": ["forts"] * count,
            "board": ["RFUD"] * count,
            "symbol": ["CRH4"] * count,
            "continuous_symbol": ["CR_NEAREST_5M"] * count,
            "timeframe": ["5m"] * count,
            "ts": timestamps,
            "open": closes,
            "high": [value + 0.1 for value in closes],
            "low": [value - 0.1 for value in closes],
            "close": closes,
            "volume": [100.0] * count,
            "value": [1000.0] * count,
            "currency": ["RUB"] * count,
            "active_contract": ["CRH4"] * count,
            "source_contract_symbol": ["CRH4"] * count,
            "dataset_type": ["nearest_contract"] * count,
            "adjustment_mode": ["unadjusted"] * count,
        }
    ).write_parquet(path)
    return path


def strategy_source() -> str:
    return """
globalThis.STRATEGY_METADATA = {
  name: "sweep_fixture",
  timeframe: "5m",
  schedule: "times",
  decision_times: ["10:05"],
  missing_decision_policy: "close",
  params: [{ name: "bias", param_type: "int", default: 1 }],
};
function onDecision(ctx, params) {
  return { target: Number(params.bias) > 0 ? 1 : -1, contracts: 1, reason: "bias" };
}
""".lstrip()
