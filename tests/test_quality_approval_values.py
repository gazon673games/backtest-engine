"""Quality approval requires finite fractions, including at fallback boundaries."""

import math

import pytest

from backtest_engine.core.exceptions import QualityGateError
from backtest_engine.core.quality_approval import (
    explicit_usable_decision_day_pct,
    is_session_gap_approval,
)


@pytest.mark.parametrize(
    "key", ["usable_decision_day_pct", "usable_day_pct", "min_usable_decision_day_pct"]
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.1, 1.1, True, "invalid"])
def test_invalid_explicit_quality_metric_cannot_pass_a_threshold(key, value):
    with pytest.raises(QualityGateError, match="finite fraction"):
        explicit_usable_decision_day_pct({key: value})


@pytest.mark.parametrize("value", [0.0, 0.95, 1.0, "0.98"])
def test_valid_quality_fractions_keep_their_value(value):
    assert explicit_usable_decision_day_pct({"usable_decision_day_pct": value}) == float(value)


@pytest.mark.parametrize("field", ["ratio", "maximum"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.1, 1.1, True, "invalid"])
def test_gap_approval_cannot_bypass_validation_using_an_invalid_fraction(field, value):
    payload = {
        "gap_audit": {"status": "approved", "session_aware": True, "gap_ratio": 0.0},
        "criteria": {"max_gap_ratio": 0.1},
    }
    if field == "ratio":
        payload["gap_audit"]["gap_ratio"] = value
    else:
        payload["criteria"]["max_gap_ratio"] = value
    assert is_session_gap_approval(payload) is False


def test_missing_explicit_metric_preserves_the_general_approval_path():
    assert explicit_usable_decision_day_pct({}) is None
    assert (
        is_session_gap_approval(
            {
                "gap_audit": {"status": "approved", "session_aware": True, "gap_ratio": 0.0},
                "criteria": {"max_gap_ratio": 0.1},
            }
        )
        is True
    )
