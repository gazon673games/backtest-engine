from __future__ import annotations

import math
from collections import Counter
from statistics import median
from typing import Any

import polars as pl

from backtest_engine.intraday.event.config import IntradayEventExecutionConfig
from backtest_engine.intraday.event.utils import as_datetime, frame_value_to_iso
from backtest_engine.research.performance import advanced_performance_metrics


def calculate_intraday_event_metrics(
    events: pl.DataFrame,
    trades: pl.DataFrame,
    execution: IntradayEventExecutionConfig,
) -> dict[str, Any]:
    returns = [float(value) for value in events.get_column("net_return").to_list()]
    equity_values = [float(value) for value in events.get_column("equity").to_list()]
    equity_cash_values = [float(value) for value in events.get_column("equity_cash").to_list()]
    positions = [float(value) for value in events.get_column("target_position").to_list()]
    event_count = events.height
    final_equity = equity_values[-1] if equity_values else 1.0
    min_equity_cash = min(equity_cash_values) if equity_cash_values else None
    insolvent = min_equity_cash is not None and min_equity_cash <= 0.0
    bars_per_trading_day = median_observed_bars_per_date(events)
    bars_per_year = (
        None
        if execution.trading_days_per_year is None
        else execution.trading_days_per_year * bars_per_trading_day
    )
    annualized_return = (
        None if insolvent else annualized_return_value(final_equity, event_count, bars_per_year)
    )
    return_volatility = std(returns)
    annualized_volatility = (
        None if insolvent or bars_per_year is None else return_volatility * math.sqrt(bars_per_year)
    )
    sharpe = None
    if not insolvent and bars_per_year is not None and return_volatility > 0.0:
        sharpe = mean_or_none(returns) / return_volatility * math.sqrt(bars_per_year)
    trade_metrics = trade_metrics_payload(trades)
    equity_values.insert(0, 1.0)
    drawdown = max_drawdown(equity_values)
    return {
        "start_ts": frame_value_to_iso(events, "ts", 0),
        "end_ts": frame_value_to_iso(events, "next_ts", -1),
        "bar_count": event_count + 1 if event_count > 0 else 0,
        "event_count": event_count,
        "strategy_call_count": strategy_call_count(events),
        "order_count": order_count(events),
        "filled_order_count": filled_order_count(events),
        "unfilled_order_count": unfilled_order_count(events),
        "partial_fill_count": partial_fill_count(events),
        "volume_limited_count": volume_limited_count(events),
        "volume_rejected_count": volume_rejected_count(events),
        "missing_decision_count": missing_decision_count(events),
        "missing_decision_close_count": missing_decision_close_count(events),
        "total_return": final_equity - 1.0,
        "final_equity_cash": equity_cash_values[-1] if equity_cash_values else None,
        "min_equity_cash": min_equity_cash,
        "insolvent": insolvent,
        "gross_pnl": sum_column(events, "gross_pnl"),
        "net_pnl": sum_column(events, "net_pnl"),
        "total_cost_cash": sum_column(events, "cost_cash"),
        "max_position_notional": max_column(events, "position_notional"),
        "turnover_notional": sum_column(events, "turnover_notional"),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        **advanced_performance_metrics(
            returns,
            equity_values,
            periods_per_year=bars_per_year,
            annualized_return=annualized_return,
            max_drawdown=drawdown,
        ),
        "trade_count": trades.height,
        **trade_metrics,
        "exposure": ratio(sum(1 for position in positions if position != 0.0), event_count),
        "long_exposure": ratio(sum(1 for position in positions if position > 0.0), event_count),
        "short_exposure": ratio(sum(1 for position in positions if position < 0.0), event_count),
        "turnover": sum_column(events, "turnover"),
        "total_cost": sum_column(events, "cost"),
        "commission_cost_cash": sum_column(events, "commission_cost_cash"),
        "slippage_cost_cash": sum_column(events, "slippage_cost_cash"),
        "spread_cost_cash": sum_column(events, "spread_cost_cash"),
        "roll_cost_cash": sum_column(events, "roll_cost_cash"),
        "trading_fee_cash": sum_column(events, "trading_fee_cash"),
        "funding_cost_cash": sum_column(events, "funding_cost_cash"),
        "liquidation_fee_cash": sum_column(events, "liquidation_fee_cash"),
        "liquidation_count": sum_boolean_column(events, "liquidated"),
        "max_maintenance_margin": max_column(events, "maintenance_margin"),
        "bars_per_trading_day": bars_per_trading_day,
        "bars_per_year": bars_per_year,
        "trading_days_per_year": execution.trading_days_per_year,
        "annualization_status": "configured"
        if bars_per_year is not None
        else "unavailable: trading_days_per_year missing",
    }


def trade_metrics_payload(trades: pl.DataFrame) -> dict[str, float | int | None]:
    if trades.height == 0:
        return {
            "long_trade_count": 0,
            "short_trade_count": 0,
            "win_rate": None,
            "avg_win": None,
            "avg_loss": None,
            "profit_factor": None,
            "best_trade": None,
            "worst_trade": None,
            "avg_trade_return": None,
            "avg_holding_bars": None,
        }
    net_returns = [float(value) for value in trades.get_column("net_return").to_list()]
    holding_bars = [float(value) for value in trades.get_column("holding_bars").to_list()]
    directions = [str(value) for value in trades.get_column("direction").to_list()]
    winning_returns = [value for value in net_returns if value > 0.0]
    losing_returns = [value for value in net_returns if value < 0.0]
    positive_sum = sum(winning_returns)
    negative_sum = sum(losing_returns)
    profit_factor = positive_sum / abs(negative_sum) if negative_sum < 0.0 else None
    return {
        "long_trade_count": sum(1 for direction in directions if direction == "long"),
        "short_trade_count": sum(1 for direction in directions if direction == "short"),
        "win_rate": len(winning_returns) / trades.height,
        "avg_win": mean_or_none(winning_returns),
        "avg_loss": mean_or_none(losing_returns),
        "profit_factor": profit_factor,
        "best_trade": max(net_returns),
        "worst_trade": min(net_returns),
        "avg_trade_return": mean_or_none(net_returns),
        "avg_holding_bars": mean_or_none(holding_bars),
    }


def median_observed_bars_per_date(events: pl.DataFrame) -> float:
    if events.height == 0:
        return 0.0
    counts = Counter(as_datetime(value).date() for value in events.get_column("ts").to_list())
    return float(median(counts.values()))


def annualized_return_value(
    final_equity: float,
    event_count: int,
    bars_per_year: float | None,
) -> float | None:
    if bars_per_year is None or event_count <= 0 or bars_per_year <= 0.0 or final_equity < 0:
        return None
    if final_equity == 0.0:
        return -1.0
    return final_equity ** (bars_per_year / event_count) - 1.0


def max_drawdown(equity_values: list[float]) -> float:
    if not equity_values:
        return 0.0
    peak = equity_values[0]
    max_drawdown_value = 0.0
    for value in equity_values:
        peak = max(peak, value)
        drawdown = value / peak - 1.0 if peak != 0.0 else 0.0
        max_drawdown_value = min(max_drawdown_value, drawdown)
    return max_drawdown_value


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def sum_column(frame: pl.DataFrame, column: str) -> float:
    if frame.height == 0 or column not in frame.columns:
        return 0.0
    return float(frame.get_column(column).sum())


def max_column(frame: pl.DataFrame, column: str) -> float:
    if frame.height == 0 or column not in frame.columns:
        return 0.0
    value = frame.get_column(column).max()
    return 0.0 if value is None else float(value)


def sum_boolean_column(frame: pl.DataFrame, column: str) -> int:
    if frame.height == 0 or column not in frame.columns:
        return 0
    return int(frame.get_column(column).sum())


def strategy_call_count(events: pl.DataFrame) -> int:
    if events.height == 0:
        return 0
    return int(events.get_column("strategy_called").sum())


def order_count(events: pl.DataFrame) -> int:
    return count_order_statuses(
        events,
        {
            "filled",
            "partial_filled",
            "partial",
            "partial_expired",
            "unfilled",
            "ambiguous_unfilled",
            "volume_rejected",
        },
    )


def filled_order_count(events: pl.DataFrame) -> int:
    return count_order_statuses(events, {"filled", "partial_filled", "partial", "partial_expired"})


def unfilled_order_count(events: pl.DataFrame) -> int:
    return count_order_statuses(events, {"unfilled", "ambiguous_unfilled", "volume_rejected"})


def partial_fill_count(events: pl.DataFrame) -> int:
    return count_order_statuses(events, {"partial_filled", "partial", "partial_expired"})


def volume_limited_count(events: pl.DataFrame) -> int:
    if events.height == 0 or "volume_limited" not in events.columns:
        return 0
    return int(events.get_column("volume_limited").sum())


def volume_rejected_count(events: pl.DataFrame) -> int:
    return count_order_statuses(events, {"volume_rejected"})


def missing_decision_count(events: pl.DataFrame) -> int:
    if events.height == 0 or "missing_decision" not in events.columns:
        return 0
    return int(events.get_column("missing_decision").sum())


def missing_decision_close_count(events: pl.DataFrame) -> int:
    if events.height == 0 or "signal_reason" not in events.columns:
        return 0
    return sum(
        1
        for value in events.get_column("signal_reason").to_list()
        if value == "missing_decision_close"
    )


def count_order_statuses(events: pl.DataFrame, statuses: set[str]) -> int:
    if events.height == 0 or "order_status" not in events.columns:
        return 0
    return sum(1 for value in events.get_column("order_status").to_list() if value in statuses)
