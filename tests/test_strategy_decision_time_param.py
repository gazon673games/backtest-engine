from __future__ import annotations

from datetime import time

import pytest

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.strategies.runtime import StrategyMetadata, StrategyParamSpec
from backtest_engine.workbench.strategy_contract import (
    strategy_decision_times,
    validate_workbench_strategy_metadata,
)


def metadata(**changes) -> StrategyMetadata:
    return StrategyMetadata(
        **{
            "name": "configurable_schedule",
            "timeframe": "5m",
            "schedule": "times",
            "decision_times": ("15:00",),
            "decision_time_param": "decision_time",
            "params": (StrategyParamSpec("decision_time", "str", default="15:00"),),
            **changes,
        }
    )


@pytest.mark.parametrize("value", ["25:00", "15:60", "15", "9:05", "15:00:00"])
def test_invalid_decision_time_param_fails_before_execution(value: str):
    with pytest.raises(StrategyPluginError, match="expected HH:MM"):
        strategy_decision_times(metadata(), {"decision_time": value})


@pytest.mark.parametrize(
    "changes",
    [
        {"decision_time_param": "unknown"},
        {"params": (StrategyParamSpec("decision_time", "int", default=15),)},
        {"schedule": "every-bar", "decision_times": ()},
        {"decision_times": ("14:40", "15:00")},
        {"decision_times": ("14:40",)},
    ],
)
def test_invalid_time_binding_is_rejected_when_strategy_is_saved(changes: dict):
    with pytest.raises(StrategyPluginError, match="decision_time"):
        validate_workbench_strategy_metadata(metadata(**changes))


def test_static_strategy_schedule_ignores_unrelated_params():
    static = metadata(decision_time_param=None)
    assert strategy_decision_times(static, {"decision_time": "16:00"}) == [time(15)]
