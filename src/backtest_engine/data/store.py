"""Owns canonical storage layout, hashes, schemas, lineage, and parquet indexes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.catalog import load_data_catalog


class DatasetLineage(BaseModel):
    parent_dataset_ids: list[str] = Field(default_factory=list)
    raw_source_paths: list[Path] = Field(default_factory=list)
    transform_name: str | None = None
    transform_params: dict[str, Any] = Field(default_factory=dict)
    adapter_snapshot: dict[str, Any] = Field(default_factory=dict)
    roll_config_sha256: str | None = None


class DatasetStageRecord(BaseModel):
    dataset_id: str
    stage: str
    version_id: str
    path: Path
    sha256: str | None = None
    parent_version_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetVersion(BaseModel):
    dataset_id: str
    version_id: str
    dataset_path: Path
    dataset_sha256: str
    schema_hash: str
    schema_version: int = 1
    row_count: int
    start_ts: str | None = None
    end_ts: str | None = None
    columns: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    lineage: DatasetLineage = Field(default_factory=DatasetLineage)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SchemaMigrationRecord(BaseModel):
    dataset_id: str
    from_schema_version: int
    to_schema_version: int
    migration_name: str
    migration_sha256: str | None = None
    notes: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DataStoreManifest(BaseModel):
    store_version: int = 1
    versions: list[DatasetVersion] = Field(default_factory=list)
    stages: list[DatasetStageRecord] = Field(default_factory=list)
    schema_migrations: list[SchemaMigrationRecord] = Field(default_factory=list)


def build_dataset_version(
    *,
    dataset_id: str,
    dataset_path: Path,
    lineage: DatasetLineage | None = None,
    schema_version: int = 1,
    version_id: str | None = None,
) -> DatasetVersion:
    if not dataset_path.exists():
        raise DatasetValidationError(f"Dataset path does not exist: {dataset_path}")
    summary = dataset_version_summary(dataset_path)
    dataset_sha = file_sha256(dataset_path)
    schema_hash = schema_mapping_hash(summary["schema"])
    return DatasetVersion(
        dataset_id=dataset_id,
        version_id=version_id or f"{dataset_sha[:12]}-{schema_hash[:12]}",
        dataset_path=dataset_path,
        dataset_sha256=dataset_sha,
        schema_hash=schema_hash,
        schema_version=schema_version,
        row_count=summary["row_count"],
        start_ts=summary["start_ts"],
        end_ts=summary["end_ts"],
        columns=list(summary["schema"]),
        dtypes={name: str(dtype) for name, dtype in summary["schema"].items()},
        lineage=lineage or DatasetLineage(),
    )


def build_data_store_from_catalog(
    *,
    catalog_path: Path,
    output_path: Path,
    adapter_snapshot: dict[str, Any] | None = None,
) -> DataStoreManifest:
    catalog = load_data_catalog(catalog_path)
    manifest = DataStoreManifest()
    for entry in catalog.entries:
        lineage = DatasetLineage(
            adapter_snapshot=adapter_snapshot or {},
            transform_params=entry.metadata,
        )
        manifest.versions.append(
            build_dataset_version(
                dataset_id=entry.dataset_id,
                dataset_path=entry.dataset_path,
                lineage=lineage,
            )
        )
    write_data_store_manifest(output_path, manifest)
    return manifest


def load_data_store_manifest(path: Path) -> DataStoreManifest:
    if not path.exists():
        return DataStoreManifest()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DataStoreManifest(**payload)


def write_data_store_manifest(path: Path, manifest: DataStoreManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_dataset_version(path: Path, version: DatasetVersion) -> DataStoreManifest:
    manifest = load_data_store_manifest(path)
    duplicate = any(
        existing.dataset_id == version.dataset_id and existing.version_id == version.version_id
        for existing in manifest.versions
    )
    if duplicate:
        raise DatasetValidationError(
            f"Dataset version already exists: {version.dataset_id}@{version.version_id}"
        )
    manifest.versions.append(version)
    write_data_store_manifest(path, manifest)
    return manifest


def add_schema_migration(
    path: Path,
    migration: SchemaMigrationRecord,
) -> DataStoreManifest:
    manifest = load_data_store_manifest(path)
    manifest.schema_migrations.append(migration)
    write_data_store_manifest(path, manifest)
    return manifest


def canonical_dataset_path(
    *,
    root: Path,
    dataset_id: str,
    stage: str,
    version_id: str,
    filename: str = "dataset.parquet",
) -> Path:
    return (
        root
        / safe_path_part(dataset_id)
        / safe_path_part(stage)
        / safe_path_part(version_id)
        / filename
    )


def append_stage_record(path: Path, stage: DatasetStageRecord) -> DataStoreManifest:
    manifest = load_data_store_manifest(path)
    manifest.stages.append(stage)
    write_data_store_manifest(path, manifest)
    return manifest


def parquet_metadata_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DatasetValidationError(f"Parquet path does not exist: {path}")
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    schema = parquet.schema_arrow
    return {
        "path": str(path),
        "format": "parquet",
        "num_rows": metadata.num_rows,
        "num_row_groups": metadata.num_row_groups,
        "schema": {field.name: str(field.type) for field in schema},
        "created_by": metadata.created_by,
    }


def safe_path_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def dataframe_schema_hash(frame: pl.DataFrame) -> str:
    return schema_mapping_hash(frame.schema)


def schema_mapping_hash(schema: dict[str, Any]) -> str:
    payload = {
        "columns": list(schema),
        "dtypes": {name: str(dtype) for name, dtype in schema.items()},
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    stat = path.stat()
    return _cached_file_sha256(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=64)
def _cached_file_sha256(path: str, _modified_ns: int, _file_size: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_version_summary(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".parquet":
        frame = _read_dataset_head(path)
        return {
            "row_count": frame.height,
            "start_ts": _frame_value(frame, "ts", 0),
            "end_ts": _frame_value(frame, "ts", -1),
            "schema": frame.schema,
        }
    metadata = parquet_metadata_index(path)
    schema = pl.read_parquet_schema(path)
    edges = (
        pl.scan_parquet(path)
        .select([pl.col("ts").min().alias("start_ts"), pl.col("ts").max().alias("end_ts")])
        .collect()
        if "ts" in schema
        else None
    )
    return {
        "row_count": int(metadata["num_rows"]),
        "start_ts": _summary_value(edges, "start_ts"),
        "end_ts": _summary_value(edges, "end_ts"),
        "schema": schema,
    }


def _summary_value(frame: pl.DataFrame | None, column: str) -> str | None:
    if frame is None or frame.height == 0:
        return None
    value = frame.get_column(column)[0]
    return None if value is None else str(value)


def optional_file_sha256(path: Path | None) -> str | None:
    return file_sha256(path) if path is not None and path.exists() else None


def _read_dataset_head(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    raise DatasetValidationError(f"Unsupported dataset format for data store: {path}")


def _frame_value(frame: pl.DataFrame, column: str, index: int) -> str | None:
    if frame.height == 0 or column not in frame.columns:
        return None
    value = frame.get_column(column)[index]
    return None if value is None else str(value)
