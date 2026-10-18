from __future__ import annotations

import json
import tomllib
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any


def audit_instrument_registry(path: Path) -> dict[str, Any]:
    violations: list[str] = []
    warnings: list[str] = []
    try:
        payload = read_json_or_toml(path)
    except Exception as exc:
        return {
            "status": "unreadable",
            "instrument_spec_count": None,
            "violations": [f"instrument registry unreadable: {exc}"],
            "warnings": [],
        }
    instruments = payload.get("instruments") if isinstance(payload, dict) else None
    if not isinstance(instruments, list) or not instruments:
        violations.append("instrument registry must contain a non-empty instruments list")
        instruments = []
    for index, instrument in enumerate(instruments):
        if not isinstance(instrument, dict):
            violations.append(f"instrument[{index}] must be an object")
            continue
        if not str(instrument.get("symbol", "")).strip():
            violations.append(f"instrument[{index}].symbol is required")
        prefix = f"instrument[{index}]"
        validate_positive_number(violations, instrument, "contract_multiplier", prefix)
        validate_positive_number(violations, instrument, "lot_size", f"instrument[{index}]")
        validate_optional_positive_number(violations, instrument, "tick_size", prefix)
        validate_optional_positive_number(violations, instrument, "tick_value", prefix)
        validate_optional_non_negative_number(
            violations,
            instrument,
            "initial_margin",
            prefix,
        )
        if instrument.get("tick_size") is None and instrument.get("tick_value") is None:
            warnings.append(f"instrument[{index}] has neither tick_size nor tick_value")
    return {
        "status": "approved" if not violations else "rejected",
        "instrument_spec_count": len(instruments),
        "violations": violations,
        "warnings": warnings,
    }


def audit_calendar(path: Path) -> dict[str, Any]:
    violations: list[str] = []
    warnings: list[str] = []
    try:
        payload = read_json_or_toml(path)
    except Exception as exc:
        return {
            "status": "unreadable",
            "holiday_count": None,
            "short_day_count": None,
            "violations": [f"calendar unreadable: {exc}"],
            "warnings": [],
        }
    if not isinstance(payload, dict):
        violations.append("calendar must be an object")
        payload = {}
    if "observed" in str(payload.get("calendar_type", "")) or "observed" in str(
        payload.get("trading_calendar", "")
    ):
        violations.append("Expected calendar must be independent of the checked dataset")
    for field_name in ("main_session_start", "main_session_end"):
        if field_name not in payload:
            violations.append(f"calendar.{field_name} is required")
        elif not looks_like_hhmm(str(payload[field_name])):
            violations.append(f"calendar.{field_name} must be HH:MM")
    for field_name in ("evening_session_start", "evening_session_end"):
        if field_name in payload and payload[field_name] is not None:
            if not looks_like_hhmm(str(payload[field_name])):
                violations.append(f"calendar.{field_name} must be HH:MM")
    holiday_dates = payload.get("holiday_dates", payload.get("holidays", []))
    short_day_dates = payload.get("short_day_dates", payload.get("short_days", []))
    validate_date_list(violations, holiday_dates, "holiday_dates")
    validate_date_list(violations, short_day_dates, "short_day_dates")
    if not holiday_dates:
        warnings.append("calendar has no holiday dates")
    return {
        "status": "approved" if not violations else "rejected",
        "holiday_count": len(holiday_dates) if isinstance(holiday_dates, list) else None,
        "short_day_count": len(short_day_dates) if isinstance(short_day_dates, list) else None,
        "violations": violations,
        "warnings": warnings,
    }


def read_json_or_toml(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".toml":
        with path.open("rb") as file:
            payload = tomllib.load(file)
    else:
        raise ValueError(f"unsupported file extension: {path.suffix}")
    if not isinstance(payload, dict):
        raise ValueError("expected object")
    return payload


def validate_positive_number(
    violations: list[str],
    payload: dict[str, Any],
    field_name: str,
    prefix: str,
) -> None:
    if field_name not in payload:
        violations.append(f"{prefix}.{field_name} is required")
        return
    validate_number(violations, payload[field_name], field_name, prefix, minimum=0.0, strict=True)


def validate_optional_positive_number(
    violations: list[str],
    payload: dict[str, Any],
    field_name: str,
    prefix: str,
) -> None:
    if payload.get(field_name) is None:
        return
    validate_number(violations, payload[field_name], field_name, prefix, minimum=0.0, strict=True)


def validate_optional_non_negative_number(
    violations: list[str],
    payload: dict[str, Any],
    field_name: str,
    prefix: str,
) -> None:
    if payload.get(field_name) is None:
        return
    validate_number(violations, payload[field_name], field_name, prefix, minimum=0.0, strict=False)


def validate_number(
    violations: list[str],
    raw_value: Any,
    field_name: str,
    prefix: str,
    *,
    minimum: float,
    strict: bool,
) -> None:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        violations.append(f"{prefix}.{field_name} must be numeric")
        return
    if not isfinite(value):
        violations.append(f"{prefix}.{field_name} must be finite")
    elif strict and value <= minimum:
        violations.append(f"{prefix}.{field_name} must be > {minimum}")
    elif not strict and value < minimum:
        violations.append(f"{prefix}.{field_name} must be >= {minimum}")


def validate_date_list(violations: list[str], values: Any, field_name: str) -> None:
    if not isinstance(values, list):
        violations.append(f"calendar.{field_name} must be a list")
        return
    for index, value in enumerate(values):
        raw_value = str(value)
        if len(raw_value) != 10:
            violations.append(f"calendar.{field_name}[{index}] must be YYYY-MM-DD")
            continue
        try:
            datetime.fromisoformat(raw_value)
        except ValueError:
            violations.append(f"calendar.{field_name}[{index}] must be YYYY-MM-DD")


def looks_like_hhmm(value: str) -> bool:
    if len(value) != 5 or value[2] != ":":
        return False
    hour, minute = value.split(":", 1)
    if not (hour.isdigit() and minute.isdigit()):
        return False
    return 0 <= int(hour) <= 23 and 0 <= int(minute) <= 59
