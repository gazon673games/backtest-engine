"""Recognize supported quality-approval schemas without conflating their metrics."""

from __future__ import annotations

from math import isfinite
from typing import Any

from backtest_engine.core.exceptions import QualityGateError


def explicit_usable_decision_day_pct(payload: dict[str, Any]) -> float | None:
    for key in (
        "usable_decision_day_pct",
        "usable_day_pct",
        "min_usable_decision_day_pct",
    ):
        value = payload.get(key)
        if value is not None:
            parsed = unit_interval(value)
            if parsed is None:
                raise QualityGateError(f"{key} must be a finite fraction in [0, 1]")
            return parsed
    return None


def is_session_gap_approval(payload: dict[str, Any]) -> bool:
    """Return true for the general pipeline approval emitted by market-data builds."""
    gap = payload.get("gap_audit")
    criteria = payload.get("criteria")
    if not isinstance(gap, dict) or not isinstance(criteria, dict):
        return False
    gap_status = str(gap.get("status")).lower()
    if gap.get("session_aware") is not True or gap_status not in {"approved", "warning"}:
        return False
    ratio = gap.get("gap_ratio")
    maximum = criteria.get("max_gap_ratio")
    ratio, maximum = unit_interval(ratio), unit_interval(maximum)
    return ratio is not None and maximum is not None and ratio <= maximum


def unit_interval(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) and 0.0 <= parsed <= 1.0 else None
