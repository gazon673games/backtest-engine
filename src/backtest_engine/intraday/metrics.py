from __future__ import annotations

import math
from typing import Any

import polars as pl

from backtest_engine.intraday.config import IntradayExecutionConfig
from backtest_engine.intraday.utils import (
    annualized_return,
    frame_value_to_iso,
    max_drawdown,
    mean_or_none,
    std,
)


def calculate_intraday_metrics(
    equity: pl.DataFrame,
    trades: pl.DataFrame,
    decision_points: pl.DataFrame,
    skipped_days: pl.DataFrame,
    execution: IntradayExecutionConfig,
) -> dict[str, Any]:
    returns = [float(value) for value in equity.get_column("net_return").to_list()]
    equity_values = [float(value) for value in equity.get_column("equity").to_list()]
    positions = [float(value) for value in equity.get_column("target_position").to_list()]
    interval_count = equity.height
    final_equity = equity_values[-1] if equity_values else 1.0

    annualized = annualized_return(
        final_equity,
        interval_count,
        execution.trading_days_per_year,
    )
    return_volatility = std(returns)
    annualized_volatility = (
        None
        if execution.trading_days_per_year is None
        else return_volatility * math.sqrt(execution.trading_days_per_year)
    )
    sharpe = None
    if execution.trading_days_per_year is not None and return_volatility > 0.0:
        sharpe = (
            sum(returns)
            / len(returns)
            / return_volatility
            * math.sqrt(execution.trading_days_per_year)
        )
    trade_metrics = calculate_trade_metrics(trades)
    equity_values.insert(0, 1.0)

    return {
        "start_ts": frame_value_to_iso(decision_points, "decision_ts", 0),
        "end_ts": frame_value_to_iso(decision_points, "decision_ts", -1),
        "decision_count": decision_points.height,
        "interval_count": interval_count,
        "skipped_decision_day_count": skipped_days.height,
        "total_return": final_equity - 1.0,
        "annualized_return": annualized,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity_values),
        "trade_count": trades.height,
        "win_rate": trade_metrics["win_rate"],
        "avg_win": trade_metrics["avg_win"],
        "avg_loss": trade_metrics["avg_loss"],
        "profit_factor": trade_metrics["profit_factor"],
        "exposure": sum(positions) / interval_count if interval_count > 0 else 0.0,
        "trading_days_per_year": execution.trading_days_per_year,
    }


def calculate_trade_metrics(trades: pl.DataFrame) -> dict[str, float | None]:
    if trades.height == 0:
        return {
            "win_rate": None,
            "avg_win": None,
            "avg_loss": None,
            "profit_factor": None,
        }

    net_returns = [float(value) for value in trades.get_column("net_return").to_list()]
    winning_returns = [value for value in net_returns if value > 0.0]
    losing_returns = [value for value in net_returns if value < 0.0]
    positive_sum = sum(winning_returns)
    negative_sum = sum(losing_returns)
    profit_factor = positive_sum / abs(negative_sum) if negative_sum < 0.0 else None

    return {
        "win_rate": len(winning_returns) / trades.height,
        "avg_win": mean_or_none(winning_returns),
        "avg_loss": mean_or_none(losing_returns),
        "profit_factor": profit_factor,
    }
