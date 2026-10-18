from __future__ import annotations

from datetime import date, datetime
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import ReportingError


def default_split_date(events: pl.DataFrame) -> date:
    dates = sorted(
        {value.date() if isinstance(value, datetime) else value for value in events["ts"]}
    )
    if not dates:
        raise ReportingError("cannot infer split date from empty events")
    return dates[len(dates) * 2 // 3]


def segment_metrics(prefix: str, events: pl.DataFrame) -> dict[str, Any]:
    return {
        f"{prefix}_start_ts": frame_value_to_iso(events, "ts", 0),
        f"{prefix}_end_ts": frame_value_to_iso(events, "next_ts", -1),
        f"{prefix}_event_count": events.height,
        f"{prefix}_total_return": segment_total_return(events),
        f"{prefix}_net_pnl": sum_column(events, "net_pnl"),
        f"{prefix}_cost_cash": sum_column(events, "cost_cash"),
        f"{prefix}_max_drawdown": max_drawdown(segment_equity(events)),
        f"{prefix}_unfilled_order_count": count_value(events, "order_status", "unfilled"),
    }


def segment_total_return(events: pl.DataFrame) -> float | None:
    if events.height == 0:
        return None
    equity = 1.0
    for value in events.get_column("net_return").to_list():
        equity *= 1.0 + float(value)
    return equity - 1.0


def segment_equity(events: pl.DataFrame) -> list[float]:
    equity = 1.0
    values = [equity]
    if events.height == 0:
        return values
    for value in events.get_column("net_return").to_list():
        equity *= 1.0 + float(value)
        values.append(equity)
    return values


def equity_from_returns(returns: list[float]) -> list[float]:
    equity = 1.0
    values = [equity]
    for value in returns:
        equity *= 1.0 + value
        values.append(equity)
    return values


def compound_values(returns: list[float]) -> float | None:
    if not returns:
        return None
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0


def stress_turnover(events: pl.DataFrame) -> list[float]:
    if "turnover" in events.columns:
        return [float(value) for value in events.get_column("turnover").fill_null(0.0).to_list()]
    if "turnover_notional" in events.columns and "equity_cash" in events.columns:
        return turnover_notional_rates(events)
    return [0.0] * events.height


def turnover_notional_rates(events: pl.DataFrame) -> list[float]:
    values = []
    previous_equity = None
    for row in events.select("turnover_notional", "equity_cash", "net_pnl").iter_rows(named=True):
        equity_cash = float(row["equity_cash"])
        net_pnl = float(row.get("net_pnl") or 0.0)
        base_equity = previous_equity if previous_equity is not None else equity_cash - net_pnl
        values.append(0.0 if base_equity == 0.0 else float(row["turnover_notional"]) / base_equity)
        previous_equity = equity_cash
    return values


def frame_value_to_iso(frame: pl.DataFrame, column: str, index: int) -> str | None:
    if frame.height == 0 or column not in frame.columns:
        return None
    value = frame.get_column(column)[index]
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def sum_column(frame: pl.DataFrame, column: str) -> float:
    if frame.height == 0 or column not in frame.columns:
        return 0.0
    return float(frame.get_column(column).sum())


def count_value(frame: pl.DataFrame, column: str, expected: str) -> int:
    if frame.height == 0 or column not in frame.columns:
        return 0
    return sum(1 for value in frame.get_column(column).to_list() if value == expected)


def min_column(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    value = frame.get_column(column).drop_nulls().min()
    return None if value is None else float(value)


def median_column(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    value = frame.get_column(column).drop_nulls().median()
    return None if value is None else float(value)


def max_column(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    value = frame.get_column(column).drop_nulls().max()
    return None if value is None else float(value)


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def float_values(frame: pl.DataFrame, column: str) -> list[float]:
    if frame.height == 0 or column not in frame.columns:
        return []
    return [float(value) for value in frame.get_column(column).drop_nulls().to_list()]


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = probability * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def max_drawdown(equity_values: list[float]) -> float:
    peak = equity_values[0]
    max_drawdown_value = 0.0
    for value in equity_values:
        peak = max(peak, value)
        drawdown = value / peak - 1.0 if peak != 0.0 else 0.0
        max_drawdown_value = min(max_drawdown_value, drawdown)
    return max_drawdown_value


def format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)
