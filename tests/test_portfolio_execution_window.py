from datetime import UTC, datetime, timedelta, timezone

import pytest

from backtest_engine.portfolio.multi.execution_window import apply_increase_window
from backtest_engine.portfolio.multi.strategy_config import SDKSleeveConfig


def sleeve(**window):
    return SDKSleeveConfig(
        asset_id="ETH",
        strategy_file="unused.py",
        initial_capital=250,
        max_exposure=5.25,
        **window,
    )


@pytest.mark.parametrize(
    "current,target,contracts,expected",
    [
        (0, 1, 2, (0, 0)),
        (2, 1, 3, (1, 2)),
        (-2, -1, 3, (-1, 2)),
        (2, 1, 1, (1, 1)),
        (2, 0, 0, (0, 0)),
        (2, -1, 3, (0, 0)),
        (-2, 1, 3, (0, 0)),
        (2, 1, 2, (1, 2)),
    ],
)
def test_weekend_blocks_entries_adds_and_reversals_but_allows_reductions(
    current, target, contracts, expected
):
    decision = {"target": target, "contracts": contracts, "reason": "signal"}
    apply_increase_window(
        decision,
        current,
        datetime(2025, 1, 4, tzinfo=UTC),
        sleeve(risk_increase_blocked_weekdays=[5, 6]),
    )
    assert (decision["target"], decision["contracts"]) == expected


def test_window_uses_utc_and_allows_unrestricted_exits_at_blocked_hours():
    window = sleeve(risk_increase_blocked_weekdays=[5, 6], risk_increase_allowed_hours=[12])
    at = datetime(2025, 1, 3, 15, tzinfo=timezone(timedelta(hours=3)))
    decision = {"target": 1, "contracts": 2, "reason": "entry"}
    apply_increase_window(decision, 0, at, window)
    assert decision["contracts"] == 2  # Friday 12:00 UTC, allowed.
    apply_increase_window(decision, 0, at + timedelta(hours=12), window)
    assert decision["contracts"] == 0  # Saturday 00:00 UTC, blocked.
    exit_decision = {"target": 0, "contracts": 0, "reason": "exit"}
    apply_increase_window(exit_decision, 2, at + timedelta(hours=12), window)
    assert exit_decision["reason"] == "exit"


@pytest.mark.parametrize(
    "window",
    [
        {"risk_increase_blocked_weekdays": [7]},
        {"risk_increase_blocked_weekdays": [-1]},
        {"risk_increase_allowed_hours": [24]},
        {"risk_increase_allowed_hours": [-1]},
    ],
)
def test_invalid_window_fails_at_configuration_boundary(window):
    with pytest.raises(ValueError):
        sleeve(**window)
