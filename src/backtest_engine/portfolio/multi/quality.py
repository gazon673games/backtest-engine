"""Per-leg quality approval enforcement for mixed portfolios."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from backtest_engine.core.exceptions import BacktestError
from backtest_engine.core.quality_approval import (
    explicit_usable_decision_day_pct,
    is_session_gap_approval,
)
from backtest_engine.portfolio.multi.config import MultiInstrumentLegConfig
from backtest_engine.portfolio.multi.utils import file_sha256


def evaluate_leg_quality_approval(
    leg: MultiInstrumentLegConfig,
    dataset_validation: dict[str, Any],
    from_date: date | None = None,
) -> dict[str, Any]:
    approval_path = leg.quality_approval_path
    if approval_path is None:
        if leg.quality_approval_required:
            raise BacktestError(
                f"quality approval is required but no path was provided for {leg.asset_id}"
            )
        return not_required_quality_approval()
    if not approval_path.exists():
        raise BacktestError(f"Quality approval file does not exist: {approval_path}")
    approval_payload = json.loads(approval_path.read_text(encoding="utf-8"))
    violations = quality_approval_violations(
        leg,
        dataset_validation,
        approval_payload,
        from_date=from_date,
    )
    result = {
        "status": "approved" if not violations else "rejected",
        "required": leg.quality_approval_required,
        "approval_path": str(approval_path),
        "approval_sha256": file_sha256(approval_path),
        "min_usable_decision_day_pct": leg.min_usable_decision_day_pct,
        "violations": violations,
        "source": approval_payload,
        "decision_coverage_basis": (
            "session_gap_approval"
            if explicit_usable_decision_day_pct(approval_payload) is None
            and is_session_gap_approval(approval_payload)
            else "approval_metric"
        ),
    }
    if violations:
        raise BacktestError("Quality approval failed: " + "; ".join(violations))
    return result


def not_required_quality_approval() -> dict[str, Any]:
    return {
        "status": "not_required",
        "required": False,
        "approval_path": None,
        "approval_sha256": None,
        "violations": [],
    }


def quality_approval_violations(
    leg: MultiInstrumentLegConfig,
    dataset_validation: dict[str, Any],
    approval_payload: dict[str, Any],
    *,
    from_date: date | None = None,
) -> list[str]:
    violations: list[str] = []
    approval_status = approval_payload.get("approval_status", approval_payload.get("status"))
    if approval_status is None:
        violations.append("approval_status is required")
    elif str(approval_status).lower() != "approved":
        violations.append(f"approval_status is not approved: {approval_status}")
    dataset_payload = approval_payload.get("dataset", {})
    if not isinstance(dataset_payload, dict):
        dataset_payload = {}
    append_mismatch_violation(
        violations,
        "continuous_symbol",
        dataset_validation.get("continuous_symbol"),
        approval_payload.get("continuous_symbol", dataset_payload.get("continuous_symbol")),
    )
    append_mismatch_violation(
        violations,
        "timeframe",
        dataset_validation.get("timeframe"),
        approval_payload.get("timeframe", dataset_payload.get("timeframe")),
    )
    append_sha_violation(violations, leg, approval_payload, dataset_payload)
    usable_pct = explicit_usable_decision_day_pct(approval_payload)
    if usable_pct is None and not is_session_gap_approval(approval_payload):
        violations.append("usable_decision_day_pct is required")
    elif usable_pct is not None and usable_pct < leg.min_usable_decision_day_pct:
        violations.append(
            "usable_decision_day_pct below required minimum: "
            f"{usable_pct} < {leg.min_usable_decision_day_pct}"
        )
    approved_from = approval_payload.get(
        "approved_from_date",
        approval_payload.get("recommended_start_date"),
    )
    if approved_from is not None:
        approved_date = datetime.fromisoformat(str(approved_from)).date()
        if from_date is not None and from_date < approved_date:
            violations.append(
                f"from_date {from_date.isoformat()} is before "
                f"approved_from_date {approved_date.isoformat()}"
            )
    return violations


def append_sha_violation(
    violations: list[str],
    leg: MultiInstrumentLegConfig,
    approval_payload: dict[str, Any],
    dataset_payload: dict[str, Any],
) -> None:
    approved_dataset_sha = approval_payload.get("dataset_sha256", dataset_payload.get("sha256"))
    if approved_dataset_sha is None:
        violations.append("dataset_sha256 is required")
    elif leg.dataset_path is not None:
        actual_sha = file_sha256(leg.dataset_path)
        if str(approved_dataset_sha) != actual_sha:
            violations.append("dataset_sha256 mismatch")


def append_mismatch_violation(
    violations: list[str],
    field_name: str,
    expected: Any,
    observed: Any,
) -> None:
    if observed is None:
        violations.append(f"{field_name} is required")
    elif str(expected) != str(observed):
        violations.append(f"{field_name} mismatch: dataset={expected}, approval={observed}")
