from __future__ import annotations

import math
from typing import Any

import polars as pl

from backtest_engine.core.models import ExecutionConfig, PerformanceMetrics
from backtest_engine.core.statistics import annualized_return as _annualized_return
from backtest_engine.core.statistics import max_drawdown as _max_drawdown
from backtest_engine.core.statistics import std as _std


def calculate_metrics(
    equity: pl.DataFrame,
    trades: pl.DataFrame,
    execution: ExecutionConfig,
) -> PerformanceMetrics:
    all_returns = [
        float(value) for value in equity.get_column("strategy_return_after_cost").to_list()
    ]
    equity_values = [float(value) for value in equity.get_column("equity").to_list()]
    positions = [float(value) for value in equity.get_column("position").to_list()]
    bar_count = equity.height
    interval_count = max(bar_count - 1, 0)
    returns = all_returns[1:]
    interval_positions = positions[1:]

    final_equity = equity_values[-1]
    annualized_return = _annualized_return(
        final_equity,
        interval_count,
        execution.trading_days_per_year,
    )
    daily_volatility = _std(returns)
    annualized_volatility = (
        None
        if execution.trading_days_per_year is None
        else daily_volatility * math.sqrt(execution.trading_days_per_year)
    )
    sharpe = None
    if execution.trading_days_per_year is not None and daily_volatility > 0 and returns:
        sharpe = (
            sum(returns)
            / len(returns)
            / daily_volatility
            * math.sqrt(execution.trading_days_per_year)
        )
    trade_metrics = _trade_metrics(trades)

    return PerformanceMetrics(
        start_ts=_to_iso(equity.get_column("ts")[0]),
        end_ts=_to_iso(equity.get_column("ts")[-1]),
        bar_count=bar_count,
        total_return=final_equity - 1.0,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        max_drawdown=_max_drawdown(equity_values),
        trade_count=trades.height,
        win_rate=trade_metrics["win_rate"],
        avg_win=trade_metrics["avg_win"],
        avg_loss=trade_metrics["avg_loss"],
        profit_factor=trade_metrics["profit_factor"],
        best_trade=trade_metrics["best_trade"],
        worst_trade=trade_metrics["worst_trade"],
        avg_trade_return=trade_metrics["avg_trade_return"],
        avg_holding_bars=trade_metrics["avg_holding_bars"],
        long_trade_count=trade_metrics["long_trade_count"],
        closed_by_end_of_data_count=trade_metrics["closed_by_end_of_data_count"],
        exposure=(
            sum(1 for position in interval_positions if abs(position) > 0.0) / interval_count
            if interval_count
            else 0.0
        ),
        mean_daily_return=sum(returns) / interval_count if interval_count else 0.0,
        daily_volatility=daily_volatility,
        best_day=max(returns) if returns else 0.0,
        worst_day=min(returns) if returns else 0.0,
    )


def _trade_metrics(trades: pl.DataFrame) -> dict[str, float | int | None]:
    if trades.height == 0:
        return {
            "win_rate": None,
            "avg_win": None,
            "avg_loss": None,
            "profit_factor": None,
            "best_trade": None,
            "worst_trade": None,
            "avg_trade_return": None,
            "avg_holding_bars": None,
            "long_trade_count": 0,
            "closed_by_end_of_data_count": 0,
        }

    net_returns = [float(value) for value in trades.get_column("net_return").to_list()]
    winning_returns = [value for value in net_returns if value > 0.0]
    losing_returns = [value for value in net_returns if value < 0.0]
    positive_sum = sum(winning_returns)
    negative_sum = sum(losing_returns)

    profit_factor = None
    if negative_sum < 0.0:
        profit_factor = positive_sum / abs(negative_sum)

    holding_bars = [float(value) for value in trades.get_column("holding_bars").to_list()]
    directions = [str(value) for value in trades.get_column("direction").to_list()]
    closed_by_end = [bool(value) for value in trades.get_column("closed_by_end_of_data").to_list()]

    return {
        "win_rate": len(winning_returns) / trades.height,
        "avg_win": _mean_or_none(winning_returns),
        "avg_loss": _mean_or_none(losing_returns),
        "profit_factor": profit_factor,
        "best_trade": max(net_returns),
        "worst_trade": min(net_returns),
        "avg_trade_return": sum(net_returns) / len(net_returns),
        "avg_holding_bars": sum(holding_bars) / len(holding_bars),
        "long_trade_count": sum(1 for direction in directions if direction == "long"),
        "closed_by_end_of_data_count": sum(1 for value in closed_by_end if value),
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _to_iso(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
