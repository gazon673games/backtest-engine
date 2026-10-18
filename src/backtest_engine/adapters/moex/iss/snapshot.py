"""Captures immutable raw MOEX ISS responses for reproducible datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backtest_engine.adapters.proxy import open_adapter_url
from backtest_engine.core.exceptions import DatasetValidationError


@dataclass(frozen=True)
class MoexIssSnapshotConfig:
    snapshot_output_path: Path
    audit_output_path: Path
    source_url: str | None = None
    raw_snapshot_path: Path | None = None
    registry_output_path: Path | None = None
    calendar_output_path: Path | None = None


@dataclass(frozen=True)
class MoexIssSnapshotResult:
    snapshot_output_path: Path
    audit_output_path: Path
    registry_output_path: Path | None
    calendar_output_path: Path | None
    status: str
    warning_count: int


def build_moex_iss_snapshot(config: MoexIssSnapshotConfig) -> MoexIssSnapshotResult:
    raw = _load_raw_snapshot(config)
    snapshot = {
        "source": "moex_iss",
        "source_url": config.source_url,
        "generated_at": datetime.now(UTC).isoformat(),
        "payload": raw,
    }
    audit = audit_moex_iss_payload(raw)
    _write_json(config.snapshot_output_path, snapshot)
    _write_json(config.audit_output_path, audit)
    if config.registry_output_path is not None:
        _write_json(config.registry_output_path, registry_from_iss_payload(raw))
    if config.calendar_output_path is not None:
        _write_json(config.calendar_output_path, calendar_from_iss_payload(raw))
    return MoexIssSnapshotResult(
        snapshot_output_path=config.snapshot_output_path,
        audit_output_path=config.audit_output_path,
        registry_output_path=config.registry_output_path,
        calendar_output_path=config.calendar_output_path,
        status=audit["status"],
        warning_count=len(audit["warnings"]),
    )


def audit_moex_iss_payload(payload: dict[str, Any]) -> dict[str, Any]:
    violations = []
    warnings = []
    if not isinstance(payload, dict):
        violations.append("snapshot payload must be a JSON object")
    if "securities" not in payload and "instruments" not in payload:
        violations.append("snapshot must contain securities or instruments")
    if "calendar" not in payload and "trading_dates" not in payload:
        warnings.append("snapshot has no calendar/trading_dates section")
    return {
        "status": "approved" if not violations else "rejected",
        "source": "moex_iss",
        "generated_at": datetime.now(UTC).isoformat(),
        "violations": violations,
        "warnings": warnings,
    }


def registry_from_iss_payload(payload: dict[str, Any]) -> dict[str, Any]:
    instruments = payload.get("instruments")
    if isinstance(instruments, list):
        return {
            "source": "moex_iss",
            "generated_at": datetime.now(UTC).isoformat(),
            "instruments": instruments,
        }
    securities = _iss_table_rows(payload.get("securities", {}))
    return {
        "source": "moex_iss",
        "generated_at": datetime.now(UTC).isoformat(),
        "instruments": [
            {
                "symbol": str(row.get("SECID") or row.get("secid") or "*"),
                "price_currency": str(row.get("CURRENCYID") or "RUB"),
                "pnl_currency": str(row.get("CURRENCYID") or "RUB"),
                "contract_multiplier": _float(row.get("LOTSIZE"), 1.0),
                "lot_size": 1.0,
                "tick_size": _optional_float(row.get("MINSTEP")),
            }
            for row in securities
        ],
    }


def calendar_from_iss_payload(payload: dict[str, Any]) -> dict[str, Any]:
    calendar = payload.get("calendar")
    if isinstance(calendar, dict):
        return {"source": "moex_iss", "generated_at": datetime.now(UTC).isoformat(), **calendar}
    return {
        "source": "moex_iss",
        "generated_at": datetime.now(UTC).isoformat(),
        "trading_calendar": "moex_iss",
        "trading_dates": payload.get("trading_dates", []),
        "holiday_dates": payload.get("holiday_dates", []),
        "short_day_dates": payload.get("short_day_dates", []),
    }


def _load_raw_snapshot(config: MoexIssSnapshotConfig) -> dict[str, Any]:
    if config.raw_snapshot_path is not None:
        return json.loads(config.raw_snapshot_path.read_text(encoding="utf-8"))
    if config.source_url is None:
        raise DatasetValidationError("source_url or raw_snapshot_path is required")
    with open_adapter_url("moex", config.source_url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _iss_table_rows(table: Any) -> list[dict[str, Any]]:
    if isinstance(table, list):
        return [row for row in table if isinstance(row, dict)]
    if not isinstance(table, dict):
        return []
    columns = table.get("columns", [])
    data = table.get("data", [])
    return [dict(zip(columns, row, strict=False)) for row in data]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _float(value: Any, default: float) -> float:
    return default if value is None else float(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
