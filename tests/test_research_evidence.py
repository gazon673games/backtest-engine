"""Invalid or unavailable evidence cannot certify a parameter search."""

import math
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.core.exceptions import ReportingError
from backtest_engine.research.selection_bias import aligned_return_matrix
from backtest_engine.research.validation.config import ResearchValidationConfig
from backtest_engine.research.validation.gate import evaluate_research_gate, walk_forward_row
from backtest_engine.research.validation.rolling import rolling_validation, select_candidate


def guardrails(**changes):
    return pl.DataFrame(
        [
            dict(
                run_count=2,
                requested_run_count=2,
                multiple_testing_warning=False,
                rank_instability_warning=False,
                oos_degradation_warning=False,
                negative_oos_run_share=0.0,
            )
            | changes
        ]
    )


def evidence(**changes):
    return pl.DataFrame(
        [
            dict(
                trial_count=2,
                probability_of_backtest_overfitting=0.0,
                reality_check_p_value=0.01,
                best_deflated_sharpe_probability=1.0,
            )
            | changes
        ]
    )


def decision(guards=None, stats=None):
    return evaluate_research_gate(
        ResearchValidationConfig(experiment_path=Path("synthetic"), max_warning_count=100),
        guardrails() if guards is None else guards,
        [],
        selection_bias=evidence() if stats is None else stats,
    )


def test_valid_independent_evidence_can_pass_the_gate():
    assert decision()["acceptance_status"] == "accepted"


@pytest.mark.parametrize(
    "field",
    [
        "probability_of_backtest_overfitting",
        "reality_check_p_value",
        "best_deflated_sharpe_probability",
    ],
)
@pytest.mark.parametrize("value", [None, math.nan, math.inf, -0.1, 1.1])
def test_missing_or_invalid_probability_never_passes_the_gate(field, value):
    result = decision(stats=evidence(**{field: value}))
    assert result["acceptance_status"] == "rejected"
    assert any(field in reason for reason in result["reasons"])


def test_missing_evidence_and_unevaluated_trials_are_rejected():
    assert decision(stats=pl.DataFrame())["acceptance_status"] == "rejected"
    assert decision(guards=pl.DataFrame())["acceptance_status"] == "rejected"
    assert decision(guards=guardrails(requested_run_count=3))["acceptance_status"] == "rejected"


@pytest.mark.parametrize("count", [0, None, 1.5])
def test_invalid_trial_count_cannot_bypass_multiple_testing_checks(count):
    assert decision(stats=evidence(trial_count=count))["acceptance_status"] == "rejected"


def events(returns):
    opened = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(returns))]
    return pl.DataFrame(
        {
            "ts": opened,
            "next_ts": [t + timedelta(days=1) for t in opened],
            "net_return": returns,
            "net_pnl": returns,
            "cost_cash": [0.0] * len(returns),
            "order_status": ["filled"] * len(returns),
        }
    )


@pytest.mark.parametrize("value", [None, math.nan, math.inf, -math.inf])
def test_invalid_returns_are_not_dropped_from_the_parameter_search(value):
    with pytest.raises(ReportingError, match="finite non-null"):
        aligned_return_matrix([({"run_id": "invalid"}, events([0.01, value]))])


def test_duplicate_timestamps_and_trial_ids_are_rejected():
    sample = events([0.01, 0.02])
    duplicated = pl.concat([sample.head(1), sample.head(1)])
    with pytest.raises(ReportingError, match="unique timestamps"):
        aligned_return_matrix([({"run_id": "duplicate-time"}, duplicated)])
    with pytest.raises(ReportingError, match="Duplicate research run_id"):
        aligned_return_matrix([({"run_id": "same"}, sample), ({"run_id": "same"}, sample)])


def test_different_evaluation_windows_are_not_silently_intersected():
    sample = events([0.01, 0.02, 0.03])
    with pytest.raises(ReportingError, match="identical observation timestamps"):
        aligned_return_matrix([({"run_id": "full"}, sample), ({"run_id": "short"}, sample.tail(2))])


def test_unfinished_return_cannot_select_the_training_winner():
    stable = events([0.01] * 4)
    delayed = events([0.0, 0.0, 0.0, 100.0]).with_columns(
        pl.when(pl.col("ts") == datetime(2024, 1, 4))
        .then(datetime(2024, 1, 10))
        .otherwise(pl.col("next_ts"))
        .alias("next_ts")
    )
    selected = select_candidate(
        [({"run_id": "stable"}, stable), ({"run_id": "delayed"}, delayed)],
        date(2024, 1, 1),
        date(2024, 1, 4),
        "total_return",
        available_before=date(2024, 1, 5),
    )
    assert selected[0]["run_id"] == "stable"
    assert selected[2] == pytest.approx(1.01**4 - 1)
    report = walk_forward_row({"run_id": "delayed"}, delayed, date(2024, 1, 5))
    assert report["is_total_return"] == 0.0
    assert report["is_event_count"] == 3


def test_rolling_validation_respects_the_configured_minimum_training_history():
    config = ResearchValidationConfig(experiment_path=Path("synthetic"), rolling_min_train_dates=20)
    folds, _, warnings = rolling_validation([({"run_id": "short"}, events([0.01] * 12))], config)
    assert folds.is_empty()
    assert any("not enough development dates" in message for message in warnings)
