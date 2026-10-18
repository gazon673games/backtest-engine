"""Shared risk-adjusted and tail metrics for single and multi-asset runs."""

from __future__ import annotations

import math
from typing import Any


def advanced_performance_metrics(
    returns: list[float],
    equity: list[float],
    *,
    periods_per_year: float | None,
    annualized_return: float | None,
    max_drawdown: float,
) -> dict[str, Any]:
    periodic_sharpe = sharpe_ratio(returns)
    annual_scale = (
        math.sqrt(periods_per_year)
        if periods_per_year is not None and periods_per_year > 0.0
        else None
    )
    sharpe = (
        periodic_sharpe * annual_scale
        if periodic_sharpe is not None and annual_scale is not None
        else None
    )
    confidence = (
        None
        if annual_scale is None
        else sharpe_confidence_interval(periodic_sharpe, len(returns), annual_scale)
    )
    return {
        "sortino": sortino_ratio(returns, periods_per_year),
        "calmar": safe_ratio(annualized_return, abs(max_drawdown)),
        "value_at_risk_95": tail_loss(returns, 0.05, expected=False),
        "expected_shortfall_95": tail_loss(returns, 0.05, expected=True),
        "return_skewness": skewness(returns),
        "return_excess_kurtosis": excess_kurtosis(returns),
        "max_drawdown_duration_periods": max_drawdown_duration(equity),
        "sharpe_ci_95_low": confidence[0] if confidence else None,
        "sharpe_ci_95_high": confidence[1] if confidence else None,
        "sharpe_recomputed": sharpe,
    }


def sharpe_ratio(values: list[float]) -> float | None:
    deviation = sample_std(values)
    return None if deviation <= 0.0 else mean(values) / deviation


def sortino_ratio(values: list[float], periods_per_year: float | None) -> float | None:
    if periods_per_year is None or not values or periods_per_year <= 0.0:
        return None
    downside = math.sqrt(sum(min(value, 0.0) ** 2 for value in values) / len(values))
    return None if downside <= 0.0 else mean(values) / downside * math.sqrt(periods_per_year)


def tail_loss(values: list[float], probability: float, *, expected: bool) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    tail_count = max(1, math.ceil(len(finite) * probability))
    tail = finite[:tail_count]
    return -mean(tail) if expected else -tail[-1]


def skewness(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    average = mean(values)
    deviation = sample_std(values)
    if deviation <= 0.0:
        return None
    count = len(values)
    moment = sum(((value - average) / deviation) ** 3 for value in values)
    return count / ((count - 1) * (count - 2)) * moment


def excess_kurtosis(values: list[float]) -> float | None:
    if len(values) < 4:
        return None
    average = mean(values)
    deviation = sample_std(values)
    if deviation <= 0.0:
        return None
    count = len(values)
    fourth = sum(((value - average) / deviation) ** 4 for value in values)
    first = count * (count + 1) / ((count - 1) * (count - 2) * (count - 3)) * fourth
    second = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
    return first - second


def max_drawdown_duration(equity: list[float]) -> int:
    peak = -math.inf
    duration = 0
    maximum = 0
    for value in equity:
        if value >= peak:
            peak = value
            duration = 0
        else:
            duration += 1
            maximum = max(maximum, duration)
    return maximum


def sharpe_confidence_interval(
    periodic_sharpe: float | None,
    count: int,
    annual_scale: float,
) -> tuple[float, float] | None:
    if periodic_sharpe is None or count < 2:
        return None
    error = math.sqrt((1.0 + 0.5 * periodic_sharpe**2) / count)
    return (
        (periodic_sharpe - 1.96 * error) * annual_scale,
        (periodic_sharpe + 1.96 * error) * annual_scale,
    )


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_ratio(numerator: float | None, denominator: float) -> float | None:
    return None if numerator is None or denominator <= 0.0 else numerator / denominator
