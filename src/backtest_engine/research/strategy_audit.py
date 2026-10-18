"""Differential lookahead and recursive-startup audits for strategy plugins."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, Field

from backtest_engine.core.exceptions import BacktestError
from backtest_engine.data.asset_schema import normalize_intraday_asset_frame
from backtest_engine.data.dataset import load_dataset
from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.plugins.loader import (
    generate_intraday_event_plugin_decision_from_callable,
    generate_plugin_signal,
    load_intraday_strategy_plugin,
)


class StrategyAuditConfig(BaseModel):
    strategy_file: Path
    dataset_path: Path
    mode: Literal["vectorized", "intraday_event"] = "intraday_event"
    params: dict[str, str] = Field(default_factory=dict)
    output_path: Path | None = None
    sample_count: int = Field(default=32, ge=1)
    recursive_windows: list[int] = Field(default_factory=lambda: [32, 64, 128, 256, 512])
    recursive_tolerance: float = Field(default=1e-9, ge=0.0)
    gate: bool = False


class StrategyAuditResult(BaseModel):
    output_path: Path
    status: str
    lookahead_checks: int
    lookahead_mismatches: int
    recursive_checks: int
    recursive_mismatches: int
    warnings: list[str] = Field(default_factory=list)


def audit_strategy(config: StrategyAuditConfig) -> StrategyAuditResult:
    frame = normalize_intraday_asset_frame(load_dataset(config.dataset_path)).sort("ts")
    if frame.height < 2:
        raise BacktestError("Strategy audit requires at least two market rows")
    if config.mode == "vectorized":
        lookahead = audit_vectorized_lookahead(frame, config)
        recursive = audit_vectorized_recursive(frame, config)
    else:
        lookahead = audit_event_lookahead(frame, config)
        recursive = audit_event_recursive(frame, config)
    warnings = []
    if not recursive:
        warnings.append("dataset is too short for recursive startup comparison")
    lookahead_mismatches = sum(not row["matches"] for row in lookahead)
    recursive_mismatches = sum(not row["matches"] for row in recursive)
    status = "failed" if lookahead_mismatches or recursive_mismatches else "passed"
    output_path = config.output_path or config.dataset_path.with_name("strategy_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": 1,
        "status": status,
        "strategy_file": str(config.strategy_file),
        "dataset_path": str(config.dataset_path),
        "mode": config.mode,
        "params": config.params,
        "lookahead": lookahead,
        "recursive": recursive,
        "warnings": warnings,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    result = StrategyAuditResult(
        output_path=output_path,
        status=status,
        lookahead_checks=len(lookahead),
        lookahead_mismatches=lookahead_mismatches,
        recursive_checks=len(recursive),
        recursive_mismatches=recursive_mismatches,
        warnings=warnings,
    )
    if config.gate and status != "passed":
        raise BacktestError(
            "Strategy bias audit failed: "
            f"lookahead={result.lookahead_mismatches}, "
            f"recursive={result.recursive_mismatches}; report={output_path}"
        )
    return result


def audit_vectorized_lookahead(
    frame: pl.DataFrame, config: StrategyAuditConfig
) -> list[dict[str, Any]]:
    baseline = generate_plugin_signal(frame, config.strategy_file, config.params)
    checks = []
    for index in sample_indices(frame.height, config.sample_count):
        prefix = frame.slice(0, index + 1)
        actual = generate_plugin_signal(prefix, config.strategy_file, config.params)[-1]
        expected = baseline[index]
        checks.append(check_row(index, expected, actual, config.recursive_tolerance))
    return checks


def audit_vectorized_recursive(
    frame: pl.DataFrame, config: StrategyAuditConfig
) -> list[dict[str, Any]]:
    windows = usable_windows(frame.height, config.recursive_windows)
    if len(windows) < 2:
        return []
    expected = generate_plugin_signal(frame.tail(windows[-1]), config.strategy_file, config.params)[
        -1
    ]
    return [
        check_row(
            window,
            expected,
            generate_plugin_signal(frame.tail(window), config.strategy_file, config.params)[-1],
            config.recursive_tolerance,
            key="startup_bars",
        )
        for window in windows[:-1]
    ]


def audit_event_lookahead(frame: pl.DataFrame, config: StrategyAuditConfig) -> list[dict[str, Any]]:
    exposed = event_decisions(frame, config, prefix_visible_only=False)
    causal = event_decisions(frame, config, prefix_visible_only=True)
    return [
        check_row(index, exposed[index], causal[index], config.recursive_tolerance)
        for index in sample_indices(frame.height, config.sample_count)
    ]


def audit_event_recursive(frame: pl.DataFrame, config: StrategyAuditConfig) -> list[dict[str, Any]]:
    windows = usable_windows(frame.height, config.recursive_windows)
    if len(windows) < 2:
        return []
    expected_frame = frame.tail(windows[-1])
    expected = event_decision(expected_frame, expected_frame.height - 1, config)
    return [
        check_row(
            window,
            expected,
            event_decision(frame.tail(window), window - 1, config),
            config.recursive_tolerance,
            key="startup_bars",
        )
        for window in windows[:-1]
    ]


def event_decision(
    frame: pl.DataFrame,
    index: int,
    config: StrategyAuditConfig,
) -> dict[str, Any]:
    return event_decisions(frame.slice(0, index + 1), config, prefix_visible_only=True)[-1]


def event_decisions(
    frame: pl.DataFrame,
    config: StrategyAuditConfig,
    *,
    prefix_visible_only: bool,
) -> list[dict[str, Any]]:
    strategy = load_intraday_strategy_plugin(config.strategy_file)
    decisions = []
    for index in range(frame.height):
        visible = frame.slice(0, index + 1) if prefix_visible_only else frame
        visible_index = visible.height - 1 if prefix_visible_only else index
        ctx = IntradayStrategyContext(visible, end_index=visible_index)
        decision = generate_intraday_event_plugin_decision_from_callable(
            strategy, ctx, config.params
        )
        decisions.append(comparable(asdict(decision)))
    return decisions


def check_row(
    value: int,
    expected: Any,
    actual: Any,
    tolerance: float,
    *,
    key: str = "index",
) -> dict[str, Any]:
    distance = difference(expected, actual)
    return {
        key: value,
        "matches": distance <= tolerance,
        "max_difference": distance,
        "expected": comparable(expected),
        "actual": comparable(actual),
    }


def difference(left: Any, right: Any) -> float:
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if isinstance(right, (int, float)) and not isinstance(right, bool):
            return abs(float(left) - float(right))
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return float("inf")
        return max((difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max((difference(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    return 0.0 if left == right else float("inf")


def comparable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): comparable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [comparable(item) for item in value]
    if isinstance(value, pl.Series):
        return comparable(value.to_list())
    return value


def sample_indices(height: int, count: int) -> list[int]:
    if height <= count:
        return list(range(1, height))
    return sorted({max(1, round(index * (height - 1) / count)) for index in range(1, count + 1)})


def usable_windows(height: int, raw_windows: list[int]) -> list[int]:
    return sorted({window for window in raw_windows if 1 < window <= height})
