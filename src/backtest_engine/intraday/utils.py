from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.statistics import annualized_return as annualized_return
from backtest_engine.core.statistics import max_drawdown as max_drawdown
from backtest_engine.core.statistics import std as std


def as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    msg = f"ts must contain datetime values, got {value!r}"
    raise DatasetValidationError(msg)


def to_iso(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def frame_value_to_iso(frame: pl.DataFrame, column: str, index: int) -> str | None:
    if frame.height == 0:
        return None
    return to_iso(frame.get_column(column)[index])


def validate_interval_prices(decision_close: float, next_decision_close: float) -> None:
    if decision_close == 0.0 or next_decision_close == 0.0:
        raise DatasetValidationError("decision close prices must not be zero")


def cost_rate(commission_bps: float, slippage_bps: float) -> float:
    return (commission_bps + slippage_bps) / 10_000.0


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
