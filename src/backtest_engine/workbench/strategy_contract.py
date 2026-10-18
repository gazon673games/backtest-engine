"""Validates the execution contract declared by a Workbench strategy."""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from typing import Any

from backtest_engine.core.exceptions import DatasetValidationError, StrategyPluginError
from backtest_engine.core.timeframes import parse_timeframe_minutes
from backtest_engine.strategies.plugins.loader import load_intraday_strategy_plugin
from backtest_engine.strategies.runtime import StrategyMetadata, validate_strategy_params

SCHEDULES = frozenset({"every-bar", "times"})
MISSING_POLICIES = frozenset({"carry", "close", "skip"})


def load_workbench_strategy_metadata(strategy_path: Path, params=None) -> StrategyMetadata:
    strategy = load_intraday_strategy_plugin(strategy_path, require_metadata=True, params=params)
    try:
        metadata = getattr(strategy, "__strategy_metadata__", None)
        if metadata is None:
            raise StrategyPluginError("Workbench strategy metadata is required")
        validate_workbench_strategy_metadata(metadata)
        return metadata
    finally:
        close = getattr(strategy, "close", None)
        if callable(close):
            close()


def validate_workbench_strategy_metadata(metadata: StrategyMetadata) -> None:
    if metadata.timeframe is None:
        raise StrategyPluginError("Strategy metadata must declare timeframe, for example '5m'")
    try:
        parse_timeframe_minutes(metadata.timeframe)
    except ValueError as exc:
        raise StrategyPluginError(str(exc)) from exc
    if metadata.schedule not in SCHEDULES:
        raise StrategyPluginError(
            f"Strategy schedule must be one of: {', '.join(sorted(SCHEDULES))}"
        )
    if metadata.missing_decision_policy not in MISSING_POLICIES:
        raise StrategyPluginError("Strategy missing_decision_policy must be carry, close, or skip")
    if metadata.decision_tolerance_minutes is not None:
        if metadata.decision_tolerance_minutes < 0:
            raise StrategyPluginError("Strategy decision_tolerance_minutes must be non-negative")
    if not 1 <= metadata.history_bars <= 100_000:
        raise StrategyPluginError("Strategy history_bars must be between 1 and 100000")
    if any(not column.strip() for column in metadata.data_columns):
        raise StrategyPluginError("Strategy data_columns must contain non-empty names")
    if len(metadata.data_columns) != len(set(metadata.data_columns)):
        raise StrategyPluginError("Strategy data_columns must not contain duplicates")
    if metadata.decision_time_param is not None:
        spec = next(
            (item for item in metadata.params if item.name == metadata.decision_time_param), None
        )
        if spec is None or spec.param_type != "str" or not isinstance(spec.default, str):
            raise StrategyPluginError("decision_time_param must name a string param with a default")
        if metadata.schedule != "times" or len(metadata.decision_times) != 1:
            raise StrategyPluginError("decision_time_param requires schedule=times and one time")
        if metadata.decision_times[0] != spec.default:
            raise StrategyPluginError("decision_times must match the decision_time_param default")
    parsed_times = strategy_decision_times(metadata)
    if metadata.schedule == "times" and not parsed_times:
        raise StrategyPluginError("Strategy schedule=times requires decision_times")
    if metadata.schedule == "every-bar" and parsed_times:
        raise StrategyPluginError("decision_times are not allowed with schedule=every-bar")
    if len(parsed_times) != len(set(parsed_times)):
        raise StrategyPluginError("Strategy decision_times must not contain duplicates")


def strategy_decision_times(
    metadata: StrategyMetadata, params: dict[str, Any] | None = None
) -> list[time]:
    raw_times = metadata.decision_times
    if metadata.decision_time_param is not None:
        try:
            values = validate_strategy_params(params or {}, metadata.params)
        except DatasetValidationError as exc:
            raise StrategyPluginError(f"Strategy params failed validation: {exc}") from exc
        raw_times = (str(values[metadata.decision_time_param]),)
    parsed = []
    for raw_value in raw_times:
        try:
            parsed_time = datetime.strptime(raw_value, "%H:%M").time()
            if parsed_time.strftime("%H:%M") != raw_value:
                raise ValueError("expected HH:MM")
            parsed.append(parsed_time)
        except ValueError as exc:
            raise StrategyPluginError(
                f"Invalid strategy decision time {raw_value!r}; expected HH:MM"
            ) from exc
    return parsed
