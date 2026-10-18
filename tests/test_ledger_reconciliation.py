from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from backtest_engine.accounting.ledger_frames import ledger_frame_from_snapshots
from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime
from backtest_engine.accounting.reconciliation import (
    read_statement,
    reconcile_ledger,
)


def test_golden_ledger_statement_reconciles() -> None:
    expected = read_statement(Path("tests/fixtures/golden/ledger_statement.csv"))
    actual = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1, 0, 1), datetime(2024, 1, 1, 0, 2)],
            "symbol": ["TEST", "TEST"],
            "currency": ["USD", "USD"],
            "net_cash_flow": [10.0, -5.0],
            "cash_balance": [1010.0, 1005.0],
            "equity_cash": [1010.0, 1005.0],
        }
    )

    result = reconcile_ledger(actual, expected, tolerance=1e-9)

    assert result["status"] == "reconciled"
    assert result["break_count"] == 0


def test_golden_ledger_statement_reports_value_break() -> None:
    expected = read_statement(Path("tests/fixtures/golden/ledger_statement.csv"))
    actual = expected.with_columns((pl.col("equity_cash") + 1.0).alias("equity_cash"))

    result = reconcile_ledger(actual, expected, tolerance=0.01)

    assert result["status"] == "failed"
    assert result["break_count"] == 2
    assert {item["field"] for item in result["breaks"]} == {"equity_cash"}


def test_spot_accounting_reconciles_to_immutable_golden_statement() -> None:
    runtime = BacktestLedgerRuntime(
        initial_capital=1_000.0,
        base_currency="USDT",
        start_ts=datetime(2024, 1, 1),
    )
    runtime.apply_spot_interval(
        fill_ts=datetime(2024, 1, 1, 0, 1),
        valuation_ts=datetime(2024, 1, 1, 0, 2),
        symbol="BTCUSDT",
        base_currency="BTC",
        quote_currency="USDT",
        quantity_delta=0.01,
        fill_price=50_000.0,
        mark_price=51_000.0,
        trading_fee_quote=0.5,
        additional_cost_quote=0.0,
        previous_equity_cash=1_000.0,
    )
    expected = read_statement(Path("tests/fixtures/golden/spot_ledger_statement.csv"))

    result = reconcile_ledger(
        ledger_frame_from_snapshots(runtime.snapshots),
        expected,
        tolerance=1e-9,
    )

    assert result["status"] == "reconciled"
