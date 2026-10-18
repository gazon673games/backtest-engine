from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.quality_approval import (
    explicit_usable_decision_day_pct,
    is_session_gap_approval,
)
from backtest_engine.intraday.config import IntradayDatasetConfig


def evaluate_intraday_quality(
    dataset: IntradayDatasetConfig,
    validation: dict[str, Any],
    from_date: date,
) -> dict[str, Any]:
    path = dataset.quality_approval_path
    if path is None:
        if dataset.quality_approval_required:
            raise DatasetValidationError(
                "quality approval is required for the legacy intraday runner"
            )
        return {"status": "not_required", "required": False, "violations": []}
    if not path.is_file():
        raise DatasetValidationError(f"Quality approval file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    violations = approval_violations(dataset, validation, from_date, payload)
    result = {
        "status": "approved" if not violations else "rejected",
        "required": dataset.quality_approval_required,
        "approval_path": str(path),
        "approval_sha256": file_sha256(path),
        "violations": violations,
        "source": payload,
        "decision_coverage_basis": (
            "runtime_schedule"
            if explicit_usable_decision_day_pct(payload) is None
            and is_session_gap_approval(payload)
            else "approval_metric"
        ),
    }
    if violations:
        raise DatasetValidationError("Quality approval failed: " + "; ".join(violations))
    return result


def approval_violations(
    dataset: IntradayDatasetConfig,
    validation: dict[str, Any],
    from_date: date,
    payload: dict[str, Any],
) -> list[str]:
    violations = []
    status = payload.get("approval_status", payload.get("status"))
    if str(status).lower() != "approved":
        violations.append(f"approval_status is not approved: {status}")
    dataset_payload = payload.get("dataset", {})
    if not isinstance(dataset_payload, dict):
        dataset_payload = {}
    _match(
        violations,
        "continuous_symbol",
        validation.get("continuous_symbol"),
        payload.get("continuous_symbol", dataset_payload.get("continuous_symbol")),
    )
    _match(
        violations,
        "timeframe",
        validation.get("timeframe"),
        payload.get("timeframe", dataset_payload.get("timeframe")),
    )
    approved_sha = payload.get("dataset_sha256", dataset_payload.get("sha256"))
    if approved_sha != file_sha256(dataset.path):
        violations.append("dataset_sha256 mismatch or missing")
    usable = explicit_usable_decision_day_pct(payload)
    if usable is None and not is_session_gap_approval(payload):
        violations.append("usable_decision_day_pct is required")
    elif usable is not None and usable < dataset.min_usable_decision_day_pct:
        violations.append("usable_decision_day_pct is below required minimum")
    approved_from = payload.get("approved_from_date", payload.get("recommended_start_date"))
    if approved_from and from_date < datetime.fromisoformat(str(approved_from)).date():
        violations.append("from_date is before approved_from_date")
    return violations


def enforce_runtime_decision_coverage(
    dataset: IntradayDatasetConfig,
    decision_count: int,
    skipped_count: int,
    approval: dict[str, Any],
) -> None:
    if approval.get("decision_coverage_basis") != "runtime_schedule":
        return
    expected = decision_count + skipped_count
    usable = decision_count / expected if expected else 0.0
    approval.update(
        {
            "usable_decision_day_pct": usable,
            "expected_decision_day_count": expected,
            "usable_decision_day_count": decision_count,
        }
    )
    if usable < dataset.min_usable_decision_day_pct:
        raise DatasetValidationError(
            "Runtime decision coverage failed: usable_decision_day_pct "
            f"{usable:.6f} is below required {dataset.min_usable_decision_day_pct:.6f}"
        )


def _match(
    violations: list[str],
    name: str,
    expected: Any,
    observed: Any,
) -> None:
    if str(observed) != str(expected):
        violations.append(f"{name} mismatch or missing")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
