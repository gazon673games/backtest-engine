from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime

START = datetime(2026, 8, 17, 15)


@pytest.mark.parametrize("cost", [10.0, -10.0])
def test_futures_decision_equity_includes_filled_costs_but_not_future_variation(cost: float):
    ledger = BacktestLedgerRuntime(base_currency="RUB", initial_capital=1000, start_ts=START)
    fill_ts = START + timedelta(minutes=5)
    mark_ts = START + timedelta(minutes=10)
    ledger.apply_event_cash_flow(
        fill_ts=fill_ts, ts=mark_ts, symbol="CRU6", currency="RUB", gross_pnl=200, cost_cash=cost
    )

    assert ledger.decision_equity_at(START) == 1000
    assert ledger.decision_equity_at(fill_ts) == 1000 - cost
    assert ledger.decision_equity_at(mark_ts) == 1200 - cost
    assert ledger.ledger.cash_flows[-2].ts == fill_ts
    assert ledger.ledger.cash_flows[-1].ts == mark_ts


def test_spot_decision_equity_values_inventory_at_the_known_fill_price():
    ledger = BacktestLedgerRuntime(initial_capital=1000, base_currency="USDT", start_ts=START)
    fill_ts = START + timedelta(minutes=5)
    mark_ts = START + timedelta(minutes=10)
    ledger.apply_spot_interval(
        fill_ts=fill_ts,
        valuation_ts=mark_ts,
        symbol="BTCUSDT",
        base_currency="BTC",
        quote_currency="USDT",
        quantity_delta=2,
        fill_price=100,
        mark_price=200,
        trading_fee_quote=1,
        additional_cost_quote=0,
        previous_equity_cash=1000,
    )

    assert ledger.decision_equity_at(fill_ts) == 999
    assert ledger.decision_equity_at(mark_ts) == 1199
