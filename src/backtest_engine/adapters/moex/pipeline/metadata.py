from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

import polars as pl
from continuous_futures_builder.session_calendar import expected_calendar_issues, observe_sessions

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.pipeline.historical_registry import enrich_historical_registry
from backtest_engine.adapters.moex.sessions import MOEX_SESSION_WINDOWS, session_window_payloads
from backtest_engine.data.store import file_sha256


@dataclass(frozen=True)
class MoexDatasetMetadata:
    instrument_registry_path: Path
    instrument_registry_audit_path: Path
    calendar_path: Path
    calendar_audit_path: Path


def attach_metadata_to_approval(
    approval_path: Path,
    manifest_path: Path,
    metadata: MoexDatasetMetadata,
) -> dict[str, Any]:
    approval = read_json(approval_path)
    registry_audit = read_json(metadata.instrument_registry_audit_path)
    calendar_audit = read_json(metadata.calendar_audit_path)
    approval["market_metadata"] = {
        "instrument_registry_path": str(metadata.instrument_registry_path),
        "instrument_registry_sha256": file_sha256(metadata.instrument_registry_path),
        "instrument_registry_audit": registry_audit,
        "calendar_path": str(metadata.calendar_path),
        "calendar_sha256": file_sha256(metadata.calendar_path),
        "calendar_audit": calendar_audit,
    }
    statuses = {registry_audit.get("status"), calendar_audit.get("status")}
    if statuses != {"approved"}:
        approval["approval_status"] = (
            "rejected"
            if "rejected" in statuses or approval.get("approval_status") == "rejected"
            else "warning"
        )
    write_json(approval_path, approval)
    manifest = read_json(manifest_path)
    manifest["approval_sha256"] = file_sha256(approval_path)
    manifest["instrument_registry_path"] = str(metadata.instrument_registry_path)
    manifest["calendar_path"] = str(metadata.calendar_path)
    write_json(manifest_path, manifest)
    return approval


def write_moex_dataset_metadata(
    *,
    output_root: Path,
    dataset_path: Path,
    instrument_rows: list[dict[str, Any]],
    expected_calendar_path: Path | None = None,
    from_date: str | None = None,
    till_date: str | None = None,
    request: DatasetBuildRequest | None = None,
) -> MoexDatasetMetadata:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    registry, registry_audit = instrument_registry_payload(instrument_rows)
    if request is not None:
        registry, registry_audit = enrich_historical_registry(
            request,
            reports,
            pl.read_parquet(dataset_path, columns=["ts", "active_contract"]),
            registry,
            registry_audit,
        )
    calendar, calendar_audit = calendar_payload(pl.read_parquet(dataset_path, columns=["ts"]))
    if expected_calendar_path is not None:
        calendar = read_json(expected_calendar_path)
        issues = expected_calendar_issues(
            pl.read_parquet(dataset_path, columns=["ts"]),
            calendar,
            from_date=from_date,
            till_date=till_date,
        )
        calendar_audit = {
            "status": "rejected" if issues else "approved",
            "violations": issues,
            "source_path": str(expected_calendar_path),
            "source_sha256": file_sha256(expected_calendar_path),
            "warnings": [],
        }
    paths = MoexDatasetMetadata(
        instrument_registry_path=reports / "instrument_registry.json",
        instrument_registry_audit_path=reports / "instrument_registry_audit.json",
        calendar_path=reports / "session_calendar.json",
        calendar_audit_path=reports / "session_calendar_audit.json",
    )
    calendar.setdefault("trading_days_per_year", 252)
    calendar.setdefault("annualization_basis", "adapter_reporting_convention")
    write_json(paths.instrument_registry_path, registry)
    write_json(paths.instrument_registry_audit_path, registry_audit)
    write_json(paths.calendar_path, calendar)
    write_json(paths.calendar_audit_path, calendar_audit)
    return paths


def instrument_registry_payload(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    warnings = []
    instruments = []
    for row in rows:
        symbol = str(row.get("SECID") or row.get("secid") or "").strip()
        if not symbol:
            continue
        spec = {
            "symbol": symbol,
            "price_currency": "RUB",
            "pnl_currency": "RUB",
            "contract_multiplier": number(row, ("LOTVOLUME", "LOTSIZE"), 1.0),
            "lot_size": 1.0,
            "quantity_step": 1.0,
            "minimum_quantity": 1.0,
            "tick_size": optional_number(row, ("MINSTEP",)),
            "tick_value": optional_number(row, ("STEPPRICE",)),
            "initial_margin": optional_number(row, ("INITIALMARGIN", "BUYDEPO")),
            "exchange_fee_per_contract": number(
                row,
                ("BUYSELLFEE", "SCALPERFEE"),
                0.0,
            ),
            "metadata": market_metadata(row),
        }
        for field in ("tick_size", "tick_value", "initial_margin"):
            if spec[field] is None:
                warnings.append(f"{symbol}: {field} is missing in ISS metadata")
        instruments.append(spec)
    violations = [] if instruments else ["instrument registry is empty"]
    generated_at = datetime.now(UTC).isoformat()
    return (
        {"source": "moex_iss", "generated_at": generated_at, "instruments": instruments},
        {
            "status": "rejected" if violations else "warning" if warnings else "approved",
            "generated_at": generated_at,
            "instrument_count": len(instruments),
            "violations": violations,
            "warnings": warnings,
        },
    )


def calendar_payload(frame: pl.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamps = frame.sort("ts").get_column("ts").to_list() if frame.height else []
    trading_dates = sorted({value.date() for value in timestamps})
    observed = observe_sessions(timestamps, main_start=time(6, 50), main_end=time(18, 50))
    holidays = []
    short_days = []
    generated_at = datetime.now(UTC).isoformat()
    calendar = {
        "source": "moex_iss_plus_observed",
        "generated_at": generated_at,
        "trading_calendar": "moex",
        "trading_dates": [value.isoformat() for value in trading_dates],
        **observed,
        "main_session_start": MOEX_SESSION_WINDOWS[0][1],
        "main_session_end": "18:50",
        "evening_session_start": "19:05",
        "evening_session_end": "23:50",
        "session_windows": session_window_payloads(),
        "short_day_policy": "allow",
    }
    violations = [] if trading_dates else ["calendar has no observed trading dates"]
    audit = {
        "status": "warning" if not violations else "rejected",
        "generated_at": generated_at,
        "trading_day_count": len(trading_dates),
        "holiday_count": len(holidays),
        "short_day_count": len(short_days),
        "violations": violations,
        "warnings": ["Observed sessions do not prove completeness; supply an independent calendar"],
    }
    return calendar, audit


def number(row: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    return optional_number(row, keys) or default


def optional_number(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def market_metadata(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("SHORTNAME", "SECNAME", "ASSETCODE", "LASTTRADEDATE", "MATDATE")
    return {key.lower(): row[key] for key in keys if row.get(key) is not None}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
