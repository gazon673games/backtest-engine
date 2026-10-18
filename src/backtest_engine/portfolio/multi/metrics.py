from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

import polars as pl

from backtest_engine.accounting.ledger_runtime import amount_in_base_currency
from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig
from backtest_engine.portfolio.multi.schemas import (
    portfolio_equity_schema,
    portfolio_legs_schema,
)
from backtest_engine.portfolio.multi.utils import count_value, max_drawdown, sharpe, to_iso
from backtest_engine.research.performance import advanced_performance_metrics


def metrics(
    portfolio_equity: pl.DataFrame,
    portfolio_legs: pl.DataFrame,
    config: MultiInstrumentBacktestConfig,
) -> dict[str, Any]:
    if portfolio_equity.height == 0:
        return empty_metrics(config)
    equity = [float(value) for value in portfolio_equity.get_column("equity").to_list()]
    returns = [float(value) for value in portfolio_equity.get_column("net_return").to_list()]
    bars_per_trading_day = median_bars_per_date(portfolio_equity)
    bars_per_year = (
        None
        if config.execution.trading_days_per_year is None
        else config.execution.trading_days_per_year * bars_per_trading_day
    )
    equity.insert(0, 1.0)
    drawdown = max_drawdown(equity)
    annualized_return = annualized_return_value(equity[-1], len(returns), bars_per_year)
    return {
        "engine": "multi_instrument_event_v1",
        "asset_count": len(config.legs),
        "event_count": portfolio_equity.height,
        "leg_event_count": portfolio_legs.height,
        "initial_capital": config.portfolio.initial_capital,
        "start_ts": to_iso(portfolio_equity.get_column("ts")[0]),
        "end_ts": to_iso(portfolio_equity.get_column("ts")[-1]),
        "final_equity_cash": float(portfolio_equity.get_column("equity_cash")[-1]),
        "total_return": equity[-1] - 1.0,
        "net_pnl": float(portfolio_equity.get_column("net_pnl").sum()),
        "gross_pnl": float(portfolio_equity.get_column("gross_pnl").sum()),
        "cost_cash": float(portfolio_equity.get_column("cost_cash").sum()),
        "turnover_notional": turnover_notional(portfolio_legs, config),
        "max_gross_position_notional": float(
            portfolio_equity.get_column("gross_position_notional").max()
        ),
        "max_margin_requirement": float(
            portfolio_equity.get_column("margin_requirement").max() or 0.0
        ),
        "max_margin_usage": float(
            portfolio_equity.get_column("margin_usage").drop_nulls().max() or 0.0
        ),
        "max_leverage": float(portfolio_equity.get_column("leverage").drop_nulls().max() or 0.0),
        "max_drawdown": drawdown,
        "sharpe": sharpe(returns, bars_per_year),
        "annualized_return": annualized_return,
        **advanced_performance_metrics(
            returns,
            equity,
            periods_per_year=bars_per_year,
            annualized_return=annualized_return,
            max_drawdown=drawdown,
        ),
        "bars_per_trading_day": bars_per_trading_day,
        "bars_per_year": bars_per_year,
        "annualization_status": "configured"
        if bars_per_year is not None
        else "unavailable: trading_days_per_year missing",
        "risk_rejected_count": count_value(portfolio_equity, "risk_status", "rejected"),
        "risk_clipped_count": count_value(portfolio_equity, "risk_status", "clipped"),
    }


def empty_metrics(config: MultiInstrumentBacktestConfig) -> dict[str, Any]:
    return {
        "engine": "multi_instrument_event_v1",
        "asset_count": len(config.legs),
        "event_count": 0,
        "leg_event_count": 0,
        "initial_capital": config.portfolio.initial_capital,
        "final_equity_cash": config.portfolio.initial_capital,
        "total_return": 0.0,
        "max_drawdown": 0.0,
    }


def turnover_notional(
    portfolio_legs: pl.DataFrame,
    config: MultiInstrumentBacktestConfig,
) -> float:
    if portfolio_legs.height == 0:
        return 0.0
    return sum(
        amount_in_base_currency(
            float(row["turnover_notional"]),
            str(row["pnl_currency"]),
            config.portfolio.base_currency,
            config.portfolio.fx_rates,
        )
        for row in portfolio_legs.select("turnover_notional", "pnl_currency").iter_rows(named=True)
    )


def portfolio_legs_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=portfolio_legs_schema(), orient="row")


def portfolio_equity_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=portfolio_equity_schema(), orient="row")


def median_bars_per_date(portfolio_equity: pl.DataFrame) -> float:
    counts = Counter(value.date() for value in portfolio_equity.get_column("ts").to_list())
    return float(median(counts.values()))


def annualized_return_value(
    final_equity: float,
    count: int,
    periods_per_year: float | None,
) -> float | None:
    if periods_per_year is None or final_equity < 0.0 or count <= 0 or periods_per_year <= 0.0:
        return None
    if final_equity == 0.0:
        return -1.0
    return final_equity ** (periods_per_year / count) - 1.0
