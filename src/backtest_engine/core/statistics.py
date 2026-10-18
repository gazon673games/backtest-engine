"""Shared return and sample statistics across engine modes."""

import math


def annualized_return(
    final_equity: float,
    interval_count: int,
    trading_days_per_year: int | None,
) -> float | None:
    if trading_days_per_year is None or interval_count <= 0 or final_equity < 0:
        return None
    if final_equity == 0:
        return -1.0
    return final_equity ** (trading_days_per_year / interval_count) - 1.0


def max_drawdown(equity_values: list[float]) -> float:
    if not equity_values:
        return 0.0
    peak = equity_values[0]
    max_drawdown_value = 0.0
    for value in equity_values:
        peak = max(peak, value)
        drawdown = value / peak - 1.0 if peak != 0 else 0.0
        max_drawdown_value = min(max_drawdown_value, drawdown)
    return max_drawdown_value


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)
