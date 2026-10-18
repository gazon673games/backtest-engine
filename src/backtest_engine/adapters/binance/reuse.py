"""Reuse a verified finalized prefix while preserving its original provenance."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from backtest_engine.adapters.binance.prefix_validation import prefix_bounds
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.timeframes import parse_timeframe_minutes
from backtest_engine.data.store import file_sha256

IDENTITY_FIELDS = (
    "engine",
    "security",
    "timeframe",
    "data_kind",
    "history_source",
    "archive_fallback_policy",
)


@dataclass(frozen=True)
class FinalizedPrefix:
    path: Path
    first_ms: int
    last_ms: int
    next_ms: int
    provenance: dict[str, Any]

    def frames(self) -> Iterator[pl.DataFrame]:
        first = datetime.fromtimestamp(self.first_ms / 1000, UTC)
        last = datetime.fromtimestamp(self.last_ms / 1000, UTC)
        for batch in pq.ParquetFile(self.path).iter_batches(batch_size=65_536):
            frame = pl.from_arrow(batch).filter(pl.col("ts").is_between(first, last))
            if not frame.is_empty():
                yield frame


def finalized_prefix(
    request: DatasetBuildRequest, storage: Path, metadata: dict[str, Any], end_ms: int
) -> FinalizedPrefix | None:
    if request.cache_policy == "network" or request.data_kind != "bars":
        return None
    if request.reuse_dataset is not None:
        return catalog_prefix(request, metadata, end_ms)
    candidates = []
    for manifest_path in storage.glob("build-*/dataset_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = manifest.get("request", {})
        if any(previous.get(name) != getattr(request, name) for name in IDENTITY_FIELDS):
            continue
        if previous.get("from_date", "9999") > request.from_date:
            continue
        validation = manifest.get("validation", {})
        if validation.get("bar_time_semantics") != "open_with_available_at":
            continue
        if validation.get("missing_mark_bars", 0) or validation.get("invalid_mark_bars", 0):
            continue
        candidates.append((validation.get("max_ts", ""), manifest_path, manifest))
    for _, manifest_path, manifest in sorted(candidates, key=lambda item: item[0], reverse=True):
        prefix = verified_prefix(request, manifest_path, manifest, metadata, end_ms)
        if prefix is not None:
            return prefix
    return None


def verified_prefix(
    request: DatasetBuildRequest,
    manifest_path: Path,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    end_ms: int,
) -> FinalizedPrefix | None:
    files = manifest["files"]
    dataset = manifest_path.parent / "market_data.parquet"
    for name, path in (("dataset", dataset), ("raw", Path(files["raw"]["path"]))):
        if not path.is_file() or file_sha256(path) != files[name]["sha256"]:
            raise DatasetValidationError(f"Binance reusable {name} hash mismatch: {path}")
    bounds = prefix_bounds(dataset, request, metadata, end_ms)
    if bounds is None:
        return None
    first_ms, last_ms, next_ms = bounds
    return FinalizedPrefix(
        dataset,
        first_ms,
        last_ms,
        next_ms,
        {
            "source": "finalized_dataset_prefix",
            "dataset_path": str(dataset.resolve()),
            "dataset_sha256": files["dataset"]["sha256"],
            "raw_snapshot_path": files["raw"]["path"],
            "raw_sha256": files["raw"]["sha256"],
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
            "from_bar": datetime.fromtimestamp(first_ms / 1000, UTC).isoformat(),
            "through_bar": datetime.fromtimestamp(last_ms / 1000, UTC).isoformat(),
            "policy": "preserve finalized observations; download only the missing suffix",
        },
    )


def catalog_prefix(
    request: DatasetBuildRequest, metadata: dict[str, Any], end_ms: int
) -> FinalizedPrefix:
    item = request.reuse_dataset
    assert item is not None
    if (
        item.adapter_id != request.adapter_id
        or item.engine != request.engine
        or item.symbol != request.security
        or parse_timeframe_minutes(item.timeframe) != parse_timeframe_minutes(request.timeframe)
    ):
        raise DatasetValidationError(f"Reusable dataset identity mismatch: {item.dataset_id}")
    if not item.dataset_path.is_file() or file_sha256(item.dataset_path) != item.dataset_sha256:
        raise DatasetValidationError(f"Binance reusable dataset hash mismatch: {item.dataset_path}")
    verify_catalog_manifest(request)
    bounds = prefix_bounds(item.dataset_path, request, metadata, end_ms)
    if bounds is None:
        raise DatasetValidationError(
            f"Reusable dataset {item.dataset_id} has incompatible or incomplete finalized bars"
        )
    first_ms, last_ms, next_ms = bounds
    return FinalizedPrefix(
        item.dataset_path,
        first_ms,
        last_ms,
        next_ms,
        {
            "source": "catalog_dataset_prefix",
            "dataset_id": item.dataset_id,
            "dataset_path": str(item.dataset_path.resolve()),
            "dataset_sha256": item.dataset_sha256,
            "quality_status": item.quality_status,
            "original_provenance": item.provenance,
            "from_bar": datetime.fromtimestamp(first_ms / 1000, UTC).isoformat(),
            "through_bar": datetime.fromtimestamp(last_ms / 1000, UTC).isoformat(),
            "policy": "preserve finalized observations; download only the missing suffix",
        },
    )


def verify_catalog_manifest(request: DatasetBuildRequest) -> None:
    item = request.reuse_dataset
    assert item is not None
    if not item.provenance.get("manifest_path"):
        return
    manifest_path = Path(item.provenance["manifest_path"])
    if not manifest_path.is_file() or file_sha256(manifest_path) != item.provenance.get(
        "manifest_sha256"
    ):
        raise DatasetValidationError(f"Reusable dataset manifest hash mismatch: {manifest_path}")
    files = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    if files["dataset"]["sha256"] != item.dataset_sha256:
        raise DatasetValidationError(f"Reusable dataset manifest differs: {item.dataset_id}")
    for name in ("raw", "quality_approval", "instrument_registry_audit"):
        path = Path(files[name]["path"])
        if not path.is_file() or file_sha256(path) != files[name]["sha256"]:
            raise DatasetValidationError(f"Binance reusable {name} hash mismatch: {path}")
    approval = json.loads(Path(files["quality_approval"]["path"]).read_text(encoding="utf-8"))
    if approval["approval_status"] != item.quality_status:
        raise DatasetValidationError(f"Reusable dataset approval differs: {item.dataset_id}")
