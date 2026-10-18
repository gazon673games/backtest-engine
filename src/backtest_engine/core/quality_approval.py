"""Recognize supported quality-approval schemas without conflating their metrics."""

from __future__ import annotations

from typing import Any


def explicit_usable_decision_day_pct(payload: dict[str, Any]) -> float | None:
    for key in (
        "usable_decision_day_pct",
        "usable_day_pct",
        "min_usable_decision_day_pct",
    ):
        value = payload.get(key)
        if value is not None:
            return float(value)
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
    if ratio is None or maximum is None:
        return False
    return float(ratio) <= float(maximum)
