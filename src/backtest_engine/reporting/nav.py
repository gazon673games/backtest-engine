"""Drawdown from ledger-derived NAV, including the starting capital."""

from math import isfinite

import polars as pl

from backtest_engine.core.exceptions import ReportingError


def initial_nav(frame: pl.DataFrame, column: str, initial_capital: float | None = None) -> float:
    if column == "equity":
        value = 1.0
    elif initial_capital is not None:
        value = initial_capital
    elif "equity" in frame.columns and float(frame["equity"][0]) != 0.0:
        value = float(frame[column][0]) / float(frame["equity"][0])
    elif "net_pnl" in frame.columns:
        value = float(frame[column][0]) - float(frame["net_pnl"][0])
    else:
        raise ReportingError("Cash NAV drawdown requires initial_capital or normalized equity")
    if not isfinite(value) or value <= 0.0:
        raise ReportingError("Drawdown starting NAV must be finite and positive")
    return value


def drawdown_values(values: list[float], initial_value: float) -> list[float]:
    peak = initial_value
    result = []
    for value in values:
        peak = max(peak, value)
        result.append(value / peak - 1.0)
    return result
