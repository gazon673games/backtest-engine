from __future__ import annotations

from pathlib import Path
from typing import Any

from backtest_engine.workbench.report import build_workbench_report, read_json


def compare_workbench_runs(output_root: Path, run_ids: list[str]) -> dict[str, Any]:
    runs = []
    for run_id in run_ids:
        output_dir = output_root / run_id
        if not output_dir.exists():
            raise FileNotFoundError(f"run not found: {run_id}")
        config = read_json(output_dir / "config.json")
        dataset_path = Path(config.get("dataset", {}).get("path", ""))
        report = build_workbench_report(output_dir, dataset_path=dataset_path, max_points=2000)
        runs.append(
            {
                "run_id": run_id,
                "metrics": report.get("metrics", {}),
                "config": report.get("config", {}),
                "equity": report.get("series", {}).get("equity", []),
                "decisions": report.get("tables", {}).get("decision_actions", [])[:50],
            }
        )
    return {"runs": runs, "metric_diff": metric_diff(runs)}


def metric_diff(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) < 2:
        return {}
    left = runs[0]["metrics"]
    right = runs[1]["metrics"]
    keys = sorted(set(left).intersection(right))
    diff = {}
    for key in keys:
        if isinstance(left.get(key), int | float) and isinstance(right.get(key), int | float):
            diff[key] = float(right[key]) - float(left[key])
    return diff
