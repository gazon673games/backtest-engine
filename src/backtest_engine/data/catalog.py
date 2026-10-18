"""Binds stable dataset identifiers to approved files and metadata artifacts."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backtest_engine.core.exceptions import DatasetValidationError


class DataCatalogEntry(BaseModel):
    dataset_id: str
    dataset_path: Path
    dataset_sha256: str | None = None
    dataset_type: str | None = None
    exchange: str | None = None
    engine: str | None = None
    market: str | None = None
    continuous_symbol: str | None = None
    timeframe: str | None = None
    adjustment_mode: str | None = None
    start_ts: str | None = None
    end_ts: str | None = None
    row_count: int | None = None
    quality_status: str | None = None
    approved_for_research: bool | None = None
    quality_approval_path: Path | None = None
    instrument_registry_path: Path | None = None
    instrument_registry_audit_path: Path | None = None
    calendar_path: Path | None = None
    calendar_audit_path: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataCatalog(BaseModel):
    catalog_version: int = 1
    entries: list[DataCatalogEntry]


def load_data_catalog(path: Path) -> DataCatalog:
    if not path.exists():
        raise DatasetValidationError(f"Data catalog does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".toml":
        with path.open("rb") as file:
            payload = tomllib.load(file)
    else:
        raise DatasetValidationError(f"Data catalog must be JSON or TOML: {path}")
    if isinstance(payload, dict) and "entries" in payload:
        return DataCatalog(**payload)
    if isinstance(payload, list):
        return DataCatalog(entries=payload)
    raise DatasetValidationError("Data catalog must contain an entries list")


def resolve_data_catalog_entry(catalog_path: Path, dataset_id: str) -> DataCatalogEntry:
    catalog = load_data_catalog(catalog_path)
    matches = [entry for entry in catalog.entries if entry.dataset_id == dataset_id]
    if not matches:
        available = ", ".join(sorted(entry.dataset_id for entry in catalog.entries)[:20])
        raise DatasetValidationError(
            f"Dataset id not found in catalog: {dataset_id}. Available: {available}"
        )
    if len(matches) > 1:
        raise DatasetValidationError(f"Dataset id is ambiguous in catalog: {dataset_id}")
    entry = matches[0]
    if not entry.dataset_path.exists():
        raise DatasetValidationError(f"Catalog dataset path does not exist: {entry.dataset_path}")
    return entry


def validate_catalog_binding(
    *,
    entry: DataCatalogEntry,
    dataset_path: Path,
    enforcement: str,
    quality_approval_path: Path | None,
    instrument_registry_path: Path | None,
    calendar_path: Path | None,
) -> dict[str, Any]:
    normalized = enforcement.strip().lower()
    if normalized not in {"off", "audit", "strict"}:
        raise DatasetValidationError(
            f"Invalid catalog enforcement: {enforcement}. Expected off, audit, or strict"
        )
    if normalized == "off":
        return _catalog_result(entry, normalized, [])

    violations: list[str] = []
    if _safe_resolve(entry.dataset_path) != _safe_resolve(dataset_path):
        violations.append(
            f"dataset_path mismatch: catalog={entry.dataset_path}, run={dataset_path}"
        )
    if entry.dataset_sha256:
        actual_sha = _file_sha256(dataset_path)
        if actual_sha != entry.dataset_sha256:
            violations.append("dataset_sha256 mismatch")
    if normalized == "strict":
        if not entry.dataset_sha256:
            violations.append("dataset_sha256 is required by strict catalog enforcement")
        if quality_approval_path is None:
            violations.append("quality_approval_path is required by strict catalog enforcement")
        elif not quality_approval_path.exists():
            violations.append(f"quality_approval_path does not exist: {quality_approval_path}")
        if instrument_registry_path is None:
            violations.append("instrument_registry_path is required by strict catalog enforcement")
        elif not instrument_registry_path.exists():
            violations.append(
                f"instrument_registry_path does not exist: {instrument_registry_path}"
            )
        if calendar_path is None:
            violations.append("calendar_path is required by strict catalog enforcement")
        elif not calendar_path.exists():
            violations.append(f"calendar_path does not exist: {calendar_path}")

    if violations:
        raise DatasetValidationError("Catalog enforcement failed: " + "; ".join(violations))
    return _catalog_result(entry, normalized, violations)


def _catalog_result(
    entry: DataCatalogEntry,
    enforcement: str,
    violations: list[str],
) -> dict[str, Any]:
    return {
        "status": "approved" if not violations else "rejected",
        "enforcement": enforcement,
        "dataset_id": entry.dataset_id,
        "dataset_path": str(entry.dataset_path),
        "dataset_sha256": entry.dataset_sha256,
        "quality_status": entry.quality_status,
        "approved_for_research": entry.approved_for_research,
        "quality_approval_path": (
            str(entry.quality_approval_path) if entry.quality_approval_path is not None else None
        ),
        "instrument_registry_path": (
            str(entry.instrument_registry_path)
            if entry.instrument_registry_path is not None
            else None
        ),
        "instrument_registry_audit_path": (
            str(entry.instrument_registry_audit_path)
            if entry.instrument_registry_audit_path is not None
            else None
        ),
        "calendar_path": str(entry.calendar_path) if entry.calendar_path is not None else None,
        "calendar_audit_path": (
            str(entry.calendar_audit_path) if entry.calendar_audit_path is not None else None
        ),
        "violations": violations,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path
