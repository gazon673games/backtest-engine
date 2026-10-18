"""Builds roll diagnostics and the formal MOEX dataset approval payload."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl


def roll_audit(frame: pl.DataFrame, *, max_gap_pct: float) -> dict[str, Any]:
    prepared = (
        frame.sort("ts")
        .select("ts", "active_contract", "close")
        .with_columns(
            [
                pl.col("active_contract").shift(1).alias("from_contract"),
                pl.col("close").shift(1).alias("from_close"),
            ]
        )
    )
    roll_rows = prepared.filter(
        pl.col("from_contract").is_not_null()
        & (pl.col("active_contract") != pl.col("from_contract"))
    )
    events = []
    for row in roll_rows.iter_rows(named=True):
        from_close = float(row["from_close"])
        to_close = float(row["close"])
        gap_pct = to_close / from_close - 1.0 if from_close > 0.0 else float("inf")
        events.append(
            {
                "ts": row["ts"].isoformat(),
                "from_contract": str(row["from_contract"]),
                "to_contract": str(row["active_contract"]),
                "from_close": from_close,
                "to_close": to_close,
                "gap_pct": gap_pct,
                "absolute_gap_pct": abs(gap_pct),
            }
        )
    max_observed = max((event["absolute_gap_pct"] for event in events), default=0.0)
    approved = max_observed <= max_gap_pct
    return {
        "status": "approved" if approved else "warning",
        "roll_count": len(events),
        "contract_count": frame.get_column("active_contract").n_unique(),
        "max_allowed_gap_pct": max_gap_pct,
        "max_observed_gap_pct": max_observed,
        "events": events[:100],
    }


def approval_payload(
    *,
    gap_result: dict[str, Any],
    roll_result: dict[str, Any],
    repair_result: dict[str, Any] | None = None,
    gap_policy: str,
    roll_gap_policy: str,
    max_gap_ratio: float,
    max_roll_gap_pct: float,
) -> dict[str, Any]:
    gap_ratio = float(gap_result.get("gap_ratio", 0.0))
    unresolved_gaps = int(gap_result["gap_count"])
    gap_accepted = unresolved_gaps == 0 or (gap_policy != "reject" and gap_ratio <= max_gap_ratio)
    roll_accepted = (
        str(roll_result.get("status")) == "approved"
        and float(roll_result.get("max_observed_gap_pct", 0.0)) <= max_roll_gap_pct
    )
    violations = []
    if not gap_accepted:
        violations.append(
            f"gap_ratio {gap_ratio:.6g} exceeds policy={gap_policy} limit {max_gap_ratio:.6g}"
        )
    if not roll_accepted:
        violations.append(f"roll gap exceeds policy={roll_gap_policy} limit {max_roll_gap_pct:.6g}")
    return {
        "approval_status": "rejected" if violations else "approved",
        "generated_at": datetime.now(UTC).isoformat(),
        "gap_policy": gap_policy,
        "roll_gap_policy": roll_gap_policy,
        "criteria": {
            "max_gap_ratio": max_gap_ratio,
            "max_roll_gap_pct": max_roll_gap_pct,
        },
        "violations": violations,
        "gap_audit": gap_result,
        "gap_repair": repair_result or {},
        "roll_audit": roll_result,
    }
