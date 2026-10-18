from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backtest_engine.adapters.raw_json import read_gzip_bytes
from backtest_engine.workbench.raw_compaction import compact_workbench_raw


def test_compaction_compresses_current_raw_and_updates_manifests(tmp_path: Path) -> None:
    moex_raw = tmp_path / "datasets" / "moex" / "raw" / "snapshot.json"
    moex_raw.parent.mkdir(parents=True)
    moex_raw.write_text('{"rows":[' + "1," * 10_000 + "1]}\n", encoding="utf-8")
    moex_manifest = moex_raw.parent.parent / "reports" / "pipeline_manifest.json"
    moex_manifest.parent.mkdir()
    write_manifest(
        moex_manifest,
        {
            "raw_snapshot_path": str(moex_raw),
            "raw_snapshot_sha256": sha256(moex_raw),
        },
    )

    binance_raw = tmp_path / "datasets" / "binance" / "adapter_raw.json"
    binance_raw.parent.mkdir()
    binance_raw.write_text('{"klines":[' + "2," * 10_000 + "2]}\n", encoding="utf-8")
    binance_manifest = binance_raw.parent / "dataset_manifest.json"
    write_manifest(
        binance_manifest,
        {"files": {"raw": {"path": str(binance_raw), "sha256": sha256(binance_raw)}}},
    )

    dry_run = compact_workbench_raw(tmp_path)
    assert len(dry_run.candidates) == 2
    assert moex_raw.exists() and binance_raw.exists()

    report = compact_workbench_raw(tmp_path, apply=True)

    assert report.saved_bytes > 0
    for source, manifest in (
        (moex_raw, moex_manifest),
        (binance_raw, binance_manifest),
    ):
        target = source.with_suffix(".json.gz")
        assert not source.exists()
        assert read_gzip_bytes(target)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        raw_entry = (
            payload["files"]["raw"]
            if "files" in payload
            else {
                "path": payload["raw_snapshot_path"],
                "sha256": payload["raw_snapshot_sha256"],
            }
        )
        assert Path(raw_entry["path"]).name == target.name
        assert raw_entry["sha256"] == sha256(target)
    assert not compact_workbench_raw(tmp_path).candidates


def test_compaction_validates_all_checksums_before_writing(tmp_path: Path) -> None:
    raw_paths = []
    for name, checksum in (("first", None), ("second", "incorrect")):
        raw_path = tmp_path / "datasets" / name / "raw" / "snapshot.json"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_text('{"rows":[1]}\n', encoding="utf-8")
        manifest = raw_path.parent.parent / "reports" / "pipeline_manifest.json"
        manifest.parent.mkdir()
        write_manifest(
            manifest,
            {
                "raw_snapshot_path": str(raw_path),
                "raw_snapshot_sha256": checksum or sha256(raw_path),
            },
        )
        raw_paths.append(raw_path)

    with pytest.raises(ValueError, match="checksum mismatch"):
        compact_workbench_raw(tmp_path, apply=True)

    assert all(path.exists() for path in raw_paths)
    assert not list(tmp_path.rglob("*.json.gz"))


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
