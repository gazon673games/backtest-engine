from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.quality_approval import (
    explicit_usable_decision_day_pct,
    is_session_gap_approval,
)
from backtest_engine.data.bar_availability import row_available_at, row_open_ts
from backtest_engine.execution.model import event_loop_stop
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig
from backtest_engine.intraday.event.row_source import FrameRows
from backtest_engine.intraday.event.schedule import build_call_schedule, iter_trading_date_indices
from backtest_engine.intraday.event.utils import file_sha256


def quality_approval_violations(
    config: IntradayEventBacktestConfig,
    dataset_validation: dict[str, Any],
    approval_payload: dict[str, Any],
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
    append_quality_sha_violation(violations, config, approval_payload, dataset_payload)
    append_quality_threshold_violations(violations, config, approval_payload)
    return violations


def append_quality_sha_violation(
    violations: list[str],
    config: IntradayEventBacktestConfig,
    approval_payload: dict[str, Any],
    dataset_payload: dict[str, Any],
) -> None:
    approved_dataset_sha = approval_payload.get("dataset_sha256", dataset_payload.get("sha256"))
    if approved_dataset_sha is None:
        violations.append("dataset_sha256 is required")
        return
    if str(approved_dataset_sha) != file_sha256(config.dataset.path):
        violations.append("dataset_sha256 mismatch")


def append_quality_threshold_violations(
    violations: list[str],
    config: IntradayEventBacktestConfig,
    approval_payload: dict[str, Any],
) -> None:
    usable_pct = explicit_usable_decision_day_pct(approval_payload)
    if usable_pct is None and not is_session_gap_approval(approval_payload):
        violations.append("usable_decision_day_pct is required")
    elif usable_pct is not None and usable_pct < config.dataset.min_usable_decision_day_pct:
        violations.append(
            "usable_decision_day_pct "
            f"{usable_pct} is below required {config.dataset.min_usable_decision_day_pct}"
        )
    append_recommended_start_violation(violations, config, approval_payload)
    append_decision_time_violation(violations, config, approval_payload)
    append_sma_window_violation(violations, config, approval_payload)


def enforce_runtime_decision_coverage(
    config: IntradayEventBacktestConfig,
    frame: pl.DataFrame,
    bar_minutes: int,
    approval: dict[str, Any],
) -> None:
    source = approval.get("source")
    if not isinstance(source, dict):
        return
    if explicit_usable_decision_day_pct(source) is not None:
        approval["decision_coverage_basis"] = "approval_metric"
        return
    if not is_session_gap_approval(source):
        return
    usable, expected, observed, deferred = _runtime_decision_coverage_counts(
        config,
        frame,
        bar_minutes,
    )
    approval.update(
        {
            "decision_coverage_basis": "runtime_schedule",
            "usable_decision_day_pct": usable,
            "expected_decision_count": expected,
            "usable_decision_count": observed,
            "deferred_trailing_decision_count": deferred,
        }
    )
    if usable < config.dataset.min_usable_decision_day_pct:
        raise DatasetValidationError(
            "Runtime decision coverage failed: usable_decision_day_pct "
            f"{usable:.6f} is below required {config.dataset.min_usable_decision_day_pct:.6f}"
        )


def runtime_decision_coverage(
    config: IntradayEventBacktestConfig,
    frame: pl.DataFrame,
    bar_minutes: int,
) -> tuple[float, int, int]:
    usable, expected, observed, _deferred = _runtime_decision_coverage_counts(
        config,
        frame,
        bar_minutes,
    )
    return usable, expected, observed


def _runtime_decision_coverage_counts(
    config: IntradayEventBacktestConfig,
    frame: pl.DataFrame,
    bar_minutes: int,
) -> tuple[float, int, int, int]:
    if config.execution.schedule == "every-bar":
        return 1.0, frame.height, frame.height, 0
    rows = FrameRows(frame)
    dates = [
        trading_date
        for trading_date, _ in iter_trading_date_indices(rows)
        if trading_date >= config.execution.from_date
    ]
    schedule = build_call_schedule(rows, config.execution, config.session, bar_minutes)
    observed = len(schedule.calls)
    observed_slots = {
        (row_open_ts(rows[index]).date(), decision_time)
        for index, decision_time in schedule.calls.items()
        if decision_time is not None
    }
    executable_count = event_loop_stop(len(rows), config.execution.timing)
    if executable_count == 0:
        expected = len(dates) * len(config.execution.decision_times)
        deferred = 0
    else:
        horizon = row_available_at(rows[executable_count - 1])
        expected = sum(
            (trading_date, decision_time) in observed_slots
            or _decision_not_after_horizon(trading_date, decision_time, horizon)
            for trading_date in dates
            for decision_time in config.execution.decision_times
        )
        deferred = len(dates) * len(config.execution.decision_times) - expected
    usable = observed / expected if expected else 0.0
    return usable, expected, observed, deferred


def _decision_not_after_horizon(
    trading_date: date,
    decision_time: time,
    horizon: datetime,
) -> bool:
    decision_at = datetime.combine(trading_date, decision_time)
    if horizon.tzinfo is not None:
        decision_at = decision_at.replace(tzinfo=horizon.tzinfo)
    return decision_at <= horizon


def append_recommended_start_violation(
    violations: list[str],
    config: IntradayEventBacktestConfig,
    approval_payload: dict[str, Any],
) -> None:
    recommended_start = approval_payload.get(
        "approved_from_date",
        approval_payload.get("recommended_start_date"),
    )
    if recommended_start is None:
        return
    recommended = datetime.fromisoformat(str(recommended_start)).date()
    if config.execution.from_date < recommended:
        violations.append(
            f"from_date {config.execution.from_date.isoformat()} is before "
            f"approved recommended_start_date {recommended.isoformat()}"
        )


def append_decision_time_violation(
    violations: list[str],
    config: IntradayEventBacktestConfig,
    approval_payload: dict[str, Any],
) -> None:
    decision_times = {
        value.isoformat(timespec="minutes") for value in config.execution.decision_times
    }
    approval_decision_time = approval_payload.get("decision_time")
    if (
        approval_decision_time
        and decision_times
        and str(approval_decision_time) not in decision_times
    ):
        violations.append(
            f"approval decision_time {approval_decision_time} does not match run "
            f"decision_times {', '.join(sorted(decision_times))}"
        )
    allowed_decision_times = approval_payload.get("allowed_decision_times")
    if allowed_decision_times and decision_times:
        allowed = {str(value) for value in allowed_decision_times}
        disallowed = sorted(decision_times.difference(allowed))
        if disallowed:
            violations.append("run decision_times are not approved: " + ", ".join(disallowed))


def append_sma_window_violation(
    violations: list[str],
    config: IntradayEventBacktestConfig,
    approval_payload: dict[str, Any],
) -> None:
    allowed_sma_windows = approval_payload.get("allowed_sma_window_bars")
    if not allowed_sma_windows:
        return
    allowed_windows = {int(value) for value in allowed_sma_windows}
    strategy_params = config.strategy.strategy_params
    for key in ("window_bars", "sma_window_bars"):
        if key in strategy_params and int(strategy_params[key]) not in allowed_windows:
            violations.append(f"{key} {strategy_params[key]} is not approved")
    if "min_session_bars" in strategy_params and "max_session_bars" in strategy_params:
        requested = set(
            range(
                int(strategy_params["min_session_bars"]),
                int(strategy_params["max_session_bars"]) + 1,
            )
        )
        if not requested.issubset(allowed_windows):
            violations.append(
                f"session bar range is not fully approved: {min(requested)}-{max(requested)}"
            )


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
