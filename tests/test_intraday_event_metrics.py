from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl
import pytest

from backtest_engine.intraday.event.config import IntradayEventExecutionConfig
from backtest_engine.intraday.event.metrics import calculate_intraday_event_metrics


def test_metrics_do_not_report_sharpe_after_equity_breaches_zero() -> None:
    start = datetime(2026, 7, 1, 10, 0)
    events = pl.DataFrame(
        {
            "ts": [start, start + timedelta(minutes=5)],
            "next_ts": [start + timedelta(minutes=5), start + timedelta(minutes=10)],
            "net_return": [-1.5, 0.5],
            "equity": [-0.5, -0.25],
            "equity_cash": [-500.0, -250.0],
            "target_position": [1.0, 1.0],
            "strategy_called": [True, False],
            "order_status": ["filled", "none"],
            "missing_decision": [False, False],
            "signal_reason": ["open", "carry"],
            "gross_pnl": [-1500.0, 250.0],
            "net_pnl": [-1500.0, 250.0],
            "cost_cash": [0.0, 0.0],
            "position_notional": [1000.0, 1000.0],
            "turnover_notional": [1000.0, 0.0],
            "turnover": [1.0, 0.0],
            "cost": [0.0, 0.0],
            "commission_cost_cash": [0.0, 0.0],
            "slippage_cost_cash": [0.0, 0.0],
            "spread_cost_cash": [0.0, 0.0],
            "roll_cost_cash": [0.0, 0.0],
        }
    )

    metrics = calculate_intraday_event_metrics(
        events,
        pl.DataFrame(),
        IntradayEventExecutionConfig(trading_days_per_year=252),
    )

    assert metrics["insolvent"] is True
    assert metrics["min_equity_cash"] == -500.0
    assert metrics["annualized_return"] is None
    assert metrics["annualized_volatility"] is None
    assert metrics["sharpe"] is None


def test_sharpe_uses_annualized_mean_period_return() -> None:
    start = datetime(2026, 7, 1, 10, 0)
    returns = [0.01, -0.005, 0.02]
    events = pl.DataFrame(
        {
            "ts": [start + timedelta(minutes=5 * index) for index in range(3)],
            "next_ts": [start + timedelta(minutes=5 * (index + 1)) for index in range(3)],
            "net_return": returns,
            "equity": [1.01, 1.00495, 1.025049],
            "equity_cash": [1010.0, 1004.95, 1025.049],
            "target_position": [1.0, 1.0, 1.0],
            "strategy_called": [True, False, False],
            "order_status": ["filled", "none", "none"],
            "missing_decision": [False, False, False],
            "signal_reason": ["open", "carry", "carry"],
            "gross_pnl": [10.0, -5.05, 20.099],
            "net_pnl": [10.0, -5.05, 20.099],
            "cost_cash": [0.0, 0.0, 0.0],
            "position_notional": [1000.0, 1000.0, 1000.0],
            "turnover_notional": [1000.0, 0.0, 0.0],
            "turnover": [1.0, 0.0, 0.0],
            "cost": [0.0, 0.0, 0.0],
            "commission_cost_cash": [0.0, 0.0, 0.0],
            "slippage_cost_cash": [0.0, 0.0, 0.0],
            "spread_cost_cash": [0.0, 0.0, 0.0],
            "roll_cost_cash": [0.0, 0.0, 0.0],
        }
    )

    metrics = calculate_intraday_event_metrics(
        events,
        pl.DataFrame(),
        IntradayEventExecutionConfig(trading_days_per_year=252),
    )
    sample_std = math.sqrt(sum((value - sum(returns) / 3) ** 2 for value in returns) / 2)
    expected = sum(returns) / 3 / sample_std * math.sqrt(252 * 3)

    assert metrics["sharpe"] == pytest.approx(expected)
