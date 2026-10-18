"""Verified resume and grid de-duplication for event batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backtest_engine.intraday.event.batch.parsing import file_sha256
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig
from backtest_engine.research.experiment_identity import (
    canonical_fingerprint,
    event_run_fingerprint,
)


def deduplicate_parameter_sets(
    parameter_sets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for params in parameter_sets:
        identity = canonical_fingerprint(params)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(params)
    return unique, len(parameter_sets) - len(unique)


def reusable_run_manifest(
    config: IntradayEventBacktestConfig,
    dataset_validation: dict[str, Any],
    *,
    verify_output_hashes: bool,
) -> dict[str, Any] | None:
    output_dir = config.output.root / config.output.run_id
    manifest_path = output_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("run_fingerprint") != event_run_fingerprint(config, dataset_validation):
        return None
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or "metrics.json" not in outputs:
        return None
    if verify_output_hashes and not outputs_are_intact(output_dir, outputs):
        return None
    return manifest


def outputs_are_intact(output_dir: Path, outputs: dict[str, Any]) -> bool:
    for name, record in outputs.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            return False
        expected_hash = record.get("sha256")
        path = output_dir / name
        if not isinstance(expected_hash, str) or not path.is_file():
            return False
        if file_sha256(path) != expected_hash:
            return False
    return True
