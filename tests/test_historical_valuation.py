from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from backtest_engine.accounting.historical_rates import HistoricalRateBook
from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime


def test_historical_rates_are_point_in_time_without_lookahead(tmp_path) -> None:
    path = tmp_path / "rates.parquet"
    pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1), datetime(2024, 1, 3)],
            "source_currency": ["USDT", "USDT"],
            "target_currency": ["RUB", "RUB"],
            "rate": [90.0, 95.0],
        }
    ).write_parquet(path)
    book = HistoricalRateBook.from_path(path)

    before_second_fixing = book.rates_at(datetime(2024, 1, 2))
    after_second_fixing = book.rates_at(datetime(2024, 1, 4))

    assert before_second_fixing["USDT/RUB"] == 90.0
    assert after_second_fixing["USDT/RUB"] == 95.0


def test_collateral_is_revalued_when_runtime_rates_change() -> None:
    runtime = BacktestLedgerRuntime(
        initial_capital=4_000.0,
        base_currency="USD",
        fx_rates={"BTC/USD": 40_000.0},
        start_ts=datetime(2024, 1, 1),
        collateral_currency="BTC",
        initial_collateral=0.1,
    )
    assert runtime.equity_cash == pytest.approx(4_000.0)

    runtime.update_fx_rates({"BTC/USD": 45_000.0})

    assert runtime.equity_cash == pytest.approx(4_500.0)
