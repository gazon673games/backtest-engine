"""Fail closed when research evidence is missing or not a valid probability."""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from backtest_engine.research.validation.config import ResearchValidationConfig


def evaluate_research_gate(
    config: ResearchValidationConfig,
    guardrails: pl.DataFrame,
    warnings: list[str],
    selection_bias: pl.DataFrame | None = None,
) -> dict[str, Any]:
    reasons = []
    if guardrails.is_empty():
        reasons.append("No evaluable research runs")
    if len(warnings) > config.max_warning_count:
        reasons.append(f"warning_count {len(warnings)} > {config.max_warning_count}")
    if guardrails.height > 0:
        row = guardrails.row(0, named=True)
        if row.get("requested_run_count", row.get("run_count")) != row.get("run_count"):
            reasons.append("Not all requested research runs have evaluable evidence")
        if bool(row.get("multiple_testing_warning")):
            reasons.append("multiple_testing_warning")
        if bool(row.get("rank_instability_warning")):
            reasons.append("rank_instability_warning")
        if bool(row.get("oos_degradation_warning")):
            reasons.append("oos_degradation_warning")
        negative_share = row.get("negative_oos_run_share")
        if not valid_probability(negative_share):
            reasons.append("negative_oos_run_share is missing or invalid")
        elif float(negative_share) > config.max_negative_oos_run_share:
            reasons.append(
                f"negative_oos_run_share {negative_share} > {config.max_negative_oos_run_share}"
            )
    if selection_bias is None or selection_bias.is_empty():
        reasons.append("Selection-bias evidence is missing")
    if selection_bias is not None and selection_bias.height:
        bias = selection_bias.row(0, named=True)
        trial_count = bias.get("trial_count")
        if not isinstance(trial_count, int) or isinstance(trial_count, bool) or trial_count < 1:
            reasons.append("Selection-bias trial_count is missing or invalid")
        elif trial_count > 1:
            pbo = bias.get("probability_of_backtest_overfitting")
            if not valid_probability(pbo):
                reasons.append("probability_of_backtest_overfitting is missing or invalid")
            elif float(pbo) > config.max_probability_of_backtest_overfitting:
                reasons.append(
                    "probability_of_backtest_overfitting "
                    f"{pbo} > {config.max_probability_of_backtest_overfitting}"
                )
            reality_p = bias.get("reality_check_p_value")
            if not valid_probability(reality_p):
                reasons.append("reality_check_p_value is missing or invalid")
            elif float(reality_p) > config.max_reality_check_p_value:
                reasons.append(
                    f"reality_check_p_value {reality_p} > {config.max_reality_check_p_value}"
                )
            deflated = bias.get("best_deflated_sharpe_probability")
            if not valid_probability(deflated):
                reasons.append("best_deflated_sharpe_probability is missing or invalid")
            elif float(deflated) < config.min_deflated_sharpe_probability:
                reasons.append(
                    "best_deflated_sharpe_probability "
                    f"{deflated} < {config.min_deflated_sharpe_probability}"
                )
    return {
        "gate_enabled": config.gate,
        "acceptance_status": "rejected" if reasons else "accepted",
        "reasons": reasons,
        "warning_count": len(warnings),
        "max_warning_count": config.max_warning_count,
        "max_negative_oos_run_share": config.max_negative_oos_run_share,
        "max_probability_of_backtest_overfitting": (config.max_probability_of_backtest_overfitting),
        "max_reality_check_p_value": config.max_reality_check_p_value,
        "min_deflated_sharpe_probability": config.min_deflated_sharpe_probability,
    }


def valid_probability(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
    except (TypeError, ValueError):
        return False
