from __future__ import annotations

from statistics import median as statistics_median
from typing import Any

import polars as pl

from backtest_engine.reporting.nav import initial_nav


def monthly_returns(events: pl.DataFrame) -> pl.DataFrame:
    if events.height == 0:
        return pl.DataFrame()
    rows = []
    working = events.with_columns(pl.col("ts").dt.strftime("%Y-%m").alias("month"))
    for month, frame in working.group_by("month", maintain_order=True):
        month_value = month[0] if isinstance(month, tuple) else month
        rows.append(
            {
                "month": month_value,
                "event_count": frame.height,
                "total_return": compound_return(frame, "net_return"),
                "gross_pnl": sum_column(frame, "gross_pnl"),
                "net_pnl": sum_column(frame, "net_pnl"),
                "cost_cash": sum_column(frame, "cost_cash"),
                "turnover_notional": sum_column(frame, "turnover_notional"),
            }
        )
    return pl.DataFrame(rows, orient="row")


def drawdown_periods(events: pl.DataFrame, *, initial_capital: float | None = None) -> pl.DataFrame:
    if events.height == 0:
        return pl.DataFrame()
    equity_column = "equity_cash" if "equity_cash" in events.columns else "equity"
    rows = []
    peak = initial_nav(events, equity_column, initial_capital)
    in_drawdown = False
    start_ts = None
    trough_ts = None
    trough_drawdown = 0.0
    duration = 0
    for row in events.iter_rows(named=True):
        equity = float(row[equity_column])
        ts = row["ts"]
        if equity >= peak:
            if in_drawdown:
                rows.append(_drawdown_row(start_ts, ts, trough_ts, trough_drawdown, duration))
                in_drawdown = False
                duration = 0
            peak = equity
            continue
        drawdown = equity / peak - 1.0 if peak != 0.0 else 0.0
        if not in_drawdown:
            in_drawdown = True
            start_ts = ts
            trough_ts = ts
            trough_drawdown = drawdown
            duration = 1
            continue
        duration += 1
        if drawdown < trough_drawdown:
            trough_drawdown = drawdown
            trough_ts = ts
    if in_drawdown:
        rows.append(
            _drawdown_row(
                start_ts,
                events.get_column("ts")[-1],
                trough_ts,
                trough_drawdown,
                duration,
            )
        )
    return pl.DataFrame(rows, orient="row")


def cost_breakdown(events: pl.DataFrame) -> pl.DataFrame:
    total_cost_cash = sum_column(events, "cost_cash")
    return pl.DataFrame(
        [
            {
                "commission_cost_cash": sum_column(events, "commission_cost_cash"),
                "slippage_cost_cash": sum_column(events, "slippage_cost_cash"),
                "spread_cost_cash": sum_column(events, "spread_cost_cash"),
                "roll_cost_cash": sum_column(events, "roll_cost_cash"),
                "total_cost_cash": total_cost_cash,
                "cost_to_gross_pnl_ratio": ratio(
                    total_cost_cash,
                    abs(sum_column(events, "gross_pnl")),
                ),
            }
        ],
        orient="row",
    )


def trade_distribution(trades: pl.DataFrame) -> pl.DataFrame:
    if trades.height == 0 or "direction" not in trades.columns:
        return pl.DataFrame()
    rows = []
    for direction, frame in trades.group_by("direction", maintain_order=True):
        direction_value = direction[0] if isinstance(direction, tuple) else direction
        net_returns = float_values(frame, "net_return")
        rows.append(
            {
                "direction": direction_value,
                "trade_count": frame.height,
                "win_rate": win_rate(net_returns),
                "avg_net_return": mean(net_returns),
                "median_net_return": median(net_returns),
                "avg_net_pnl": mean(float_values(frame, "net_pnl")),
                "best_trade": max(net_returns) if net_returns else None,
                "worst_trade": min(net_returns) if net_returns else None,
            }
        )
    return pl.DataFrame(rows, orient="row")


def compound_return(frame: pl.DataFrame, column: str) -> float | None:
    if frame.height == 0 or column not in frame.columns:
        return None
    equity = 1.0
    for value in frame.get_column(column).drop_nulls().to_list():
        equity *= 1.0 + float(value)
    return equity - 1.0


def sum_column(frame: pl.DataFrame, column: str) -> float:
    if frame.height == 0 or column not in frame.columns:
        return 0.0
    return float(frame.get_column(column).sum())


def float_values(frame: pl.DataFrame, column: str) -> list[float]:
    if frame.height == 0 or column not in frame.columns:
        return []
    return [float(value) for value in frame.get_column(column).drop_nulls().to_list()]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics_median(values) if values else None


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def win_rate(values: list[float]) -> float | None:
    return sum(1 for value in values if value > 0.0) / len(values) if values else None


def _drawdown_row(
    start_ts: Any,
    end_ts: Any,
    trough_ts: Any,
    trough_drawdown: float,
    duration: int,
) -> dict[str, Any]:
    return {
        "start_ts": to_iso(start_ts),
        "end_ts": to_iso(end_ts),
        "trough_ts": to_iso(trough_ts),
        "max_drawdown": trough_drawdown,
        "duration_bars": duration,
    }


def to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
