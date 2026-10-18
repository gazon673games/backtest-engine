"""Combines market-data and point-in-time rule audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.store import file_sha256


def quality_payload(
    frame: pl.DataFrame | None,
    request: DatasetBuildRequest,
    validation: dict[str, Any],
    rule_audit: dict[str, Any],
) -> dict[str, Any]:
    gap_count = (
        int(validation.get("gap_count", 0))
        if frame is None
        else (count_gaps(frame, request.timeframe) if request.data_kind == "bars" else 0)
    )
    gap_rejected = request.gap_policy == "reject" and gap_count > 0
    violations = list(rule_audit["violations"])
    if gap_rejected:
        violations.append(f"detected {gap_count} timestamp gaps")
    elif gap_count:
        violations.append(f"detected {gap_count} timestamp gaps; no repair applied")
    status = "rejected" if gap_rejected else rule_audit["status"]
    if status == "not_required":
        status = "warning" if violations else "approved"
    if gap_count and status == "approved":
        status = "warning"
    prefix = request.reuse_dataset if request.cache_policy != "network" else None
    if prefix is not None and prefix.quality_status == "warning":
        if status == "approved":
            status = "warning"
        violations.append(
            f"reused diagnostic dataset {prefix.dataset_id}; original assumptions retained"
        )
    if validation.get("missing_mark_bars", 0):
        status = "rejected"
        violations.append(
            f"missing mark prices in {validation['missing_mark_bars']} perpetual bars"
        )
    if validation.get("invalid_mark_bars", 0):
        status = "rejected"
        violations.append(
            f"invalid mark prices in {validation['invalid_mark_bars']} perpetual bars"
        )
    if validation.get("unpriced_funding_bars", 0):
        if status == "approved":
            status = "warning"
        violations.append(
            f"{validation['unpriced_funding_bars']} funding bars lack event prices; "
            "diagnostic settlement uses the bar mark approximation"
        )
    if validation.get("estimated_funding_bars", 0):
        if status == "approved":
            status = "warning"
        violations.append(
            f"{validation['estimated_funding_bars']} funding bars use opening mark estimates; "
            "these are not exchange-reported settlement prices"
        )
    row_count = validation["row_count"]
    missing = int(validation.get("missing_bars", gap_count))
    gap_ratio = missing / max(1, row_count + missing)
    return {
        "approval_status": status,
        "gap_policy": request.gap_policy,
        "gap_count": gap_count,
        "row_count": row_count,
        "continuous_symbol": validation["continuous_symbol"],
        "timeframe": validation["timeframe"],
        "dataset_sha256": file_sha256(Path(validation["path"])),
        "gap_audit": {
            "status": "approved" if gap_count == 0 else "warning",
            "session_aware": True,
            "calendar": "24x7",
            "gap_ratio": gap_ratio,
            "missing_bars": missing,
        },
        "criteria": {"max_gap_ratio": 0.0},
        "coverage": {
            "policy": "available_history",
            "requested_from": request.from_date,
            "requested_till": request.till_date,
            "first_bar": validation["min_ts"],
            "last_bar": validation["max_ts"],
        },
        "validation": validation,
        "rule_audit": rule_audit,
        "reused_dataset": prefix.model_dump(mode="json") if prefix is not None else None,
        "violations": violations,
    }


def count_gaps(frame: pl.DataFrame, timeframe: str) -> int:
    minutes = timeframe_minutes(timeframe)
    timestamps = frame.get_column("ts").to_list()
    return sum(
        1
        for previous, current in zip(timestamps, timestamps[1:], strict=False)
        if (current - previous).total_seconds() > minutes * 60
    )


def timeframe_minutes(timeframe: str) -> int:
    value = timeframe.lower()
    if value.endswith("m"):
        return int(value[:-1])
    if value.endswith("h"):
        return int(value[:-1]) * 60
    if value == "1d":
        return 1440
    raise DatasetValidationError(f"Unsupported gap audit timeframe: {timeframe}")
