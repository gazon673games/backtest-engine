from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.adapters.models import DatasetBuildRequest, DatasetBuildResult
from backtest_engine.core.file_io import file_transaction, write_json_atomic
from backtest_engine.data.catalog import DataCatalog, DataCatalogEntry
from backtest_engine.data.store import file_sha256, parquet_metadata_index


def catalog_path_for(output_root: Path) -> Path:
    return output_root.parent / "datasets" / "catalog.json"


def register_workbench_dataset(
    output_root: Path,
    request: DatasetBuildRequest,
    result: DatasetBuildResult,
) -> DatasetBuildResult:
    path = catalog_path_for(output_root)
    summary = dataset_catalog_summary(result.dataset_path)
    entry = DataCatalogEntry(
        dataset_id=result.dataset_id,
        dataset_path=result.dataset_path.resolve(),
        dataset_sha256=file_sha256(result.dataset_path),
        dataset_type=summary.get("dataset_type"),
        exchange=summary.get("exchange"),
        engine=request.engine,
        market=request.market,
        continuous_symbol=summary.get("continuous_symbol"),
        timeframe=result.timeframe,
        adjustment_mode=summary.get("adjustment_mode"),
        start_ts=summary.get("start_ts"),
        end_ts=summary.get("end_ts"),
        row_count=result.row_count,
        quality_status=result.approval_status,
        approved_for_research=result.approval_status == "approved",
        quality_approval_path=resolved(result.approval_path),
        instrument_registry_path=resolved(result.instrument_registry_path),
        instrument_registry_audit_path=resolved(result.instrument_registry_audit_path),
        calendar_path=resolved(result.calendar_path),
        calendar_audit_path=resolved(result.calendar_audit_path),
        metadata={
            "adapter_id": request.adapter_id,
            "security": request.security,
            "instrument_mode": request.instrument_mode,
            "roll_rule": request.roll_rule,
            "built_at": datetime.now(UTC).isoformat(),
            "requested_from_date": request.from_date,
            "requested_till_date": request.till_date,
            "source_as_of": result.source_as_of,
            "finalized_through": result.finalized_through,
            "source_delay_minutes": request.source_delay_minutes,
            "bar_time_semantics": result.bar_time_semantics,
            "manifest_path": str(result.manifest_path.resolve()),
        },
    )
    write_json_atomic(result.output_root / "workbench_dataset.json", entry.model_dump(mode="json"))
    upsert_catalog_entry(path, entry)
    return result.model_copy(update={"catalog_path": path.resolve()})


def dataset_catalog_summary(path: Path) -> dict[str, Any]:
    """Read catalog fields with Parquet projection instead of loading every bar."""
    schema = pl.read_parquet_schema(path)
    value_columns = [
        name
        for name in ("dataset_type", "exchange", "continuous_symbol", "adjustment_mode")
        if name in schema
    ]
    expressions = [pl.col(name).drop_nulls().first().alias(name) for name in value_columns]
    if "ts" in schema:
        expressions.extend(
            [
                pl.col("ts").min().alias("start_ts"),
                pl.col("ts").max().alias("end_ts"),
            ]
        )
    if not expressions:
        return {}
    result = pl.scan_parquet(path).select(expressions).collect().row(0, named=True)
    for name in ("start_ts", "end_ts"):
        value = result.get(name)
        if value is not None:
            result[name] = value.isoformat() if hasattr(value, "isoformat") else str(value)
    return result


def list_workbench_datasets(output_root: Path) -> dict[str, Any]:
    path = catalog_path_for(output_root)
    catalog = load_or_discover_catalog(path)
    items = [catalog_item(entry, path) for entry in catalog.entries]
    items.sort(key=lambda item: str(item.get("built_at") or ""), reverse=True)
    return {"catalog_path": str(path.resolve()), "items": items}


def get_workbench_dataset(output_root: Path, dataset_id: str) -> dict[str, Any]:
    payload = list_workbench_datasets(output_root)
    for item in payload["items"]:
        if item["dataset_id"] == dataset_id:
            return item
    raise FileNotFoundError(f"dataset not found: {dataset_id}")


def upsert_catalog_entry(path: Path, entry: DataCatalogEntry) -> None:
    with file_transaction(path):
        catalog = load_or_discover_catalog(path)
        entries = [item for item in catalog.entries if item.dataset_id != entry.dataset_id]
        entries.insert(0, entry)
        write_catalog(path, DataCatalog(entries=entries))


def load_or_discover_catalog(path: Path) -> DataCatalog:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DataCatalog.model_validate(payload)
    return DataCatalog(entries=discover_dataset_entries(path.parent))


def discover_dataset_entries(root: Path) -> list[DataCatalogEntry]:
    entries = []
    if not root.exists():
        return entries
    for declaration in root.rglob("workbench_dataset.json"):
        entries.append(DataCatalogEntry.model_validate(read_json(declaration)))
    for dataset_path in root.rglob("bars/nearest.parquet"):
        reports = dataset_path.parent.parent / "reports"
        manifest = read_json(reports / "pipeline_manifest.json")
        approval = read_json(reports / "approval.json")
        dataset = approval.get("dataset", {}) if isinstance(approval, dict) else {}
        dataset_id = str(manifest.get("dataset_id") or dataset_path.parent.parent.name)
        entries.append(
            DataCatalogEntry(
                dataset_id=dataset_id,
                dataset_path=dataset_path.resolve(),
                dataset_sha256=str(dataset.get("sha256") or file_sha256(dataset_path)),
                continuous_symbol=manifest.get("continuous_symbol"),
                timeframe=str(dataset.get("timeframe") or ""),
                row_count=int(
                    dataset.get("row_count") or parquet_metadata_index(dataset_path)["num_rows"]
                ),
                quality_status=approval.get("approval_status"),
                approved_for_research=approval.get("approval_status") == "approved",
                quality_approval_path=existing(reports / "approval.json"),
                instrument_registry_path=existing(reports / "instrument_registry.json"),
                instrument_registry_audit_path=existing(reports / "instrument_registry_audit.json"),
                calendar_path=existing(reports / "session_calendar.json"),
                calendar_audit_path=existing(reports / "session_calendar_audit.json"),
                metadata={
                    "discovered": True,
                    "manifest_path": str(reports / "pipeline_manifest.json"),
                },
            )
        )
    by_id = {}
    for entry in sorted(entries, key=lambda item: str(item.metadata.get("built_at") or "")):
        by_id[entry.dataset_id] = entry
    return list(by_id.values())


def catalog_item(entry: DataCatalogEntry, catalog_path: Path) -> dict[str, Any]:
    return {
        **entry.model_dump(mode="json"),
        "catalog_path": str(catalog_path.resolve()),
        "exists": entry.dataset_path.exists(),
        "built_at": entry.metadata.get("built_at"),
    }


def write_catalog(path: Path, catalog: DataCatalog) -> None:
    write_json_atomic(path, catalog.model_dump(mode="json"))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def resolved(path: Path | None) -> Path | None:
    return None if path is None else path.resolve()


def existing(path: Path) -> Path | None:
    return path.resolve() if path.exists() else None
