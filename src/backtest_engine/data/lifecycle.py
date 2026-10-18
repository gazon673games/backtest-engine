"""Audits whether cataloged datasets remain reproducible and internally consistent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel

from backtest_engine.data.catalog import DataCatalogEntry, load_data_catalog
from backtest_engine.data.lifecycle_checks import audit_calendar, audit_instrument_registry
from backtest_engine.data.lifecycle_markdown import markdown_table


class DataLifecycleAuditConfig(BaseModel):
    catalog_path: Path
    output_path: Path | None = None
    strict: bool = True


class DataLifecycleAuditResult(BaseModel):
    output_path: Path
    audit_path: Path
    report_path: Path
    entry_count: int
    approved_count: int
    rejected_count: int


def audit_data_lifecycle(config: DataLifecycleAuditConfig) -> DataLifecycleAuditResult:
    catalog = load_data_catalog(config.catalog_path)
    output_path = config.output_path or config.catalog_path.parent / "data_lifecycle_audit"
    output_path.mkdir(parents=True, exist_ok=True)
    rows = [_audit_entry(entry, config.strict) for entry in catalog.entries]
    audit = pl.DataFrame(rows, orient="row") if rows else _empty_audit_frame()
    audit_path = output_path / "data_lifecycle_audit.csv"
    audit.write_csv(audit_path)
    audit.write_parquet(audit_path.with_suffix(".parquet"))
    audit_path.with_suffix(".json").write_text(
        json.dumps(audit.to_dicts() if audit.height else [], indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    report_path = output_path / "data_lifecycle_report.md"
    report_path.write_text(_render_report(config, audit), encoding="utf-8")
    approved_count = _count_status(audit, "approved")
    rejected_count = _count_status(audit, "rejected")
    return DataLifecycleAuditResult(
        output_path=output_path,
        audit_path=audit_path,
        report_path=report_path,
        entry_count=audit.height,
        approved_count=approved_count,
        rejected_count=rejected_count,
    )


def _audit_entry(entry: DataCatalogEntry, strict: bool) -> dict[str, Any]:
    violations: list[str] = []
    warnings: list[str] = []
    dataset_path = entry.dataset_path
    path_exists = dataset_path.exists()
    actual_sha = None
    actual_row_count = None
    actual_start_ts = None
    actual_end_ts = None
    actual_continuous_symbol = None
    actual_timeframe = None
    actual_adjustment_mode = None
    instrument_registry_status = None
    instrument_spec_count = None
    calendar_status = None
    holiday_count = None
    short_day_count = None

    if not path_exists:
        violations.append(f"dataset_path missing: {dataset_path}")
    else:
        actual_sha = _file_sha256(dataset_path)
        if entry.dataset_sha256 and actual_sha != entry.dataset_sha256:
            violations.append("dataset_sha256 mismatch")
        try:
            frame = pl.read_parquet(dataset_path)
            actual_row_count = frame.height
            actual_start_ts = _frame_value(frame, "ts", 0)
            actual_end_ts = _frame_value(frame, "ts", -1)
            actual_continuous_symbol = _unique_value(frame, "continuous_symbol")
            actual_timeframe = _unique_value(frame, "timeframe")
            actual_adjustment_mode = _unique_value(frame, "adjustment_mode")
            _compare_declared(
                violations,
                "row_count",
                entry.row_count,
                actual_row_count,
            )
            _compare_declared(
                violations,
                "continuous_symbol",
                entry.continuous_symbol,
                actual_continuous_symbol,
            )
            _compare_declared(violations, "timeframe", entry.timeframe, actual_timeframe)
            _compare_declared(
                warnings,
                "adjustment_mode",
                entry.adjustment_mode,
                actual_adjustment_mode,
            )
        except Exception as exc:
            violations.append(f"dataset unreadable: {exc}")

    quality_status = _approval_status(entry.quality_approval_path)
    if strict:
        if entry.quality_approval_path is None:
            violations.append("quality_approval_path missing")
        elif quality_status != "approved":
            violations.append(f"quality approval is not approved: {quality_status}")
        if entry.instrument_registry_path is None or not entry.instrument_registry_path.exists():
            violations.append("instrument_registry_path missing")
            instrument_registry_status = "missing"
        else:
            registry_result = audit_instrument_registry(entry.instrument_registry_path)
            instrument_registry_status = registry_result["status"]
            instrument_spec_count = registry_result["instrument_spec_count"]
            violations.extend(registry_result["violations"])
            warnings.extend(registry_result["warnings"])
        if entry.calendar_path is None or not entry.calendar_path.exists():
            violations.append("calendar_path missing")
            calendar_status = "missing"
        else:
            calendar_result = audit_calendar(entry.calendar_path)
            calendar_status = calendar_result["status"]
            holiday_count = calendar_result["holiday_count"]
            short_day_count = calendar_result["short_day_count"]
            violations.extend(calendar_result["violations"])
            warnings.extend(calendar_result["warnings"])

    status = "approved" if not violations else "rejected"
    return {
        "dataset_id": entry.dataset_id,
        "status": status,
        "dataset_path": str(dataset_path),
        "path_exists": path_exists,
        "declared_sha256": entry.dataset_sha256,
        "actual_sha256": actual_sha,
        "sha256_match": entry.dataset_sha256 is None or actual_sha == entry.dataset_sha256,
        "declared_row_count": entry.row_count,
        "actual_row_count": actual_row_count,
        "declared_continuous_symbol": entry.continuous_symbol,
        "actual_continuous_symbol": actual_continuous_symbol,
        "declared_timeframe": entry.timeframe,
        "actual_timeframe": actual_timeframe,
        "declared_adjustment_mode": entry.adjustment_mode,
        "actual_adjustment_mode": actual_adjustment_mode,
        "actual_start_ts": actual_start_ts,
        "actual_end_ts": actual_end_ts,
        "quality_status": quality_status,
        "quality_approval_path": (
            str(entry.quality_approval_path) if entry.quality_approval_path is not None else None
        ),
        "instrument_registry_path": (
            str(entry.instrument_registry_path)
            if entry.instrument_registry_path is not None
            else None
        ),
        "instrument_registry_status": instrument_registry_status,
        "instrument_spec_count": instrument_spec_count,
        "calendar_path": str(entry.calendar_path) if entry.calendar_path is not None else None,
        "calendar_status": calendar_status,
        "holiday_count": holiday_count,
        "short_day_count": short_day_count,
        "violation_count": len(violations),
        "warning_count": len(warnings),
        "violations": "; ".join(violations),
        "warnings": "; ".join(warnings),
    }


def _render_report(config: DataLifecycleAuditConfig, audit: pl.DataFrame) -> str:
    lines = [
        "# Data Lifecycle Audit",
        "",
        f"- catalog_path: `{config.catalog_path}`",
        f"- strict: `{config.strict}`",
        f"- entry_count: {audit.height}",
        f"- approved_count: {_count_status(audit, 'approved')}",
        f"- rejected_count: {_count_status(audit, 'rejected')}",
        "",
    ]
    if audit.height == 0:
        lines.append("_No catalog entries._")
        return "\n".join(lines)
    rejected = audit.filter(pl.col("status") == "rejected")
    if rejected.height > 0:
        lines.extend(["## Rejected Entries", ""])
        for row in rejected.iter_rows(named=True):
            lines.append(f"- `{row['dataset_id']}`: {row['violations']}")
        lines.append("")
    lines.extend(["## Catalog Snapshot", "", markdown_table(audit.select(_report_columns()))])
    return "\n".join(lines)


def _approval_status(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.exists():
        return "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "unreadable"
    return str(payload.get("approval_status", payload.get("status", "unknown"))).lower()


def _compare_declared(
    problems: list[str],
    field_name: str,
    declared: Any,
    actual: Any,
) -> None:
    if declared is None or actual is None:
        return
    if str(declared) != str(actual):
        problems.append(f"{field_name} mismatch: declared={declared}, actual={actual}")


def _frame_value(frame: pl.DataFrame, column: str, index: int) -> str | None:
    if frame.height == 0 or column not in frame.columns:
        return None
    value = frame.get_column(column)[index]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _unique_value(frame: pl.DataFrame, column: str) -> str | None:
    if frame.height == 0 or column not in frame.columns:
        return None
    values = frame.get_column(column).drop_nulls().unique().to_list()
    if len(values) != 1:
        return None
    return str(values[0])


def _count_status(audit: pl.DataFrame, status: str) -> int:
    if audit.height == 0 or "status" not in audit.columns:
        return 0
    return sum(1 for value in audit.get_column("status").to_list() if value == status)


def _report_columns() -> list[str]:
    return [
        "dataset_id",
        "status",
        "actual_continuous_symbol",
        "actual_timeframe",
        "actual_row_count",
        "sha256_match",
        "quality_status",
        "instrument_registry_status",
        "instrument_spec_count",
        "calendar_status",
        "holiday_count",
        "short_day_count",
        "violation_count",
        "warning_count",
    ]


def _empty_audit_frame() -> pl.DataFrame:
    return pl.DataFrame([], schema={column: pl.String for column in _report_columns()})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
