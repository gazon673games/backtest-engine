from __future__ import annotations

import json
from pathlib import Path

from backtest_engine.workbench.storage_cleanup import cleanup_workbench_storage


def test_cleanup_removes_only_redundant_workbench_files(tmp_path: Path) -> None:
    paired_csv = tmp_path / "runs" / "run-1" / "events.csv"
    paired_csv.parent.mkdir(parents=True)
    paired_csv.write_text("ts,close\n", encoding="utf-8")
    paired_csv.with_suffix(".parquet").write_bytes(b"parquet")
    unpaired_csv = paired_csv.parent / "manual-export.csv"
    unpaired_csv.write_text("keep\n", encoding="utf-8")

    dataset = tmp_path / "datasets" / "dataset-1"
    raw_dir = dataset / "raw"
    raw_dir.mkdir(parents=True)
    current = raw_dir / "current.json"
    current.write_text("{}", encoding="utf-8")
    superseded = raw_dir / "old.json"
    superseded.write_text("{}", encoding="utf-8")
    legacy = dataset / "adapter_raw" / "candles_raw.json"
    legacy.parent.mkdir()
    legacy.write_text("{}", encoding="utf-8")
    reports = dataset / "reports"
    reports.mkdir()
    (reports / "pipeline_manifest.json").write_text(
        json.dumps({"raw_snapshot_path": str(current)}),
        encoding="utf-8",
    )

    dry_run = cleanup_workbench_storage(tmp_path)

    assert dry_run.applied is False
    assert {item.category for item in dry_run.candidates} == {
        "paired_csv",
        "superseded_raw",
        "legacy_adapter_raw",
    }
    assert paired_csv.exists() and superseded.exists() and legacy.exists()

    applied = cleanup_workbench_storage(tmp_path, apply=True)

    assert applied.total_bytes == dry_run.total_bytes
    assert not paired_csv.exists()
    assert not superseded.exists()
    assert not legacy.exists()
    assert current.exists()
    assert unpaired_csv.exists()
    assert paired_csv.with_suffix(".parquet").exists()
