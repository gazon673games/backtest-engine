"""Activates point-in-time valuation rates for each event interval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from backtest_engine.accounting.historical_rates import HistoricalRateBook
from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig


@dataclass
class EventValuation:
    rate_book: HistoricalRateBook | None
    ledger: BacktestLedgerRuntime

    @classmethod
    def create(
        cls,
        config: IntradayEventBacktestConfig,
        start_ts: datetime | None,
    ) -> EventValuation:
        path = config.portfolio.valuation_rates_path
        rate_book = HistoricalRateBook.from_path(path) if path is not None else None
        rates = (
            rate_book.rates_at(start_ts, config.portfolio.fx_rates)
            if rate_book is not None and start_ts is not None
            else config.portfolio.fx_rates
        )
        ledger = BacktestLedgerRuntime(
            initial_capital=config.portfolio.initial_capital,
            base_currency=config.portfolio.base_currency,
            fx_rates=rates,
            start_ts=start_ts,
            collateral_currency=config.portfolio.collateral_currency,
            initial_collateral=config.portfolio.initial_collateral,
            rate_book=rate_book,
        )
        return cls(rate_book=rate_book, ledger=ledger)

    def activate(
        self,
        config: IntradayEventBacktestConfig,
        ts: datetime,
    ) -> IntradayEventBacktestConfig:
        if self.rate_book is None:
            return config
        rates = self.ledger.activate_rates(ts)
        portfolio = config.portfolio.model_copy(update={"fx_rates": rates})
        return config.model_copy(update={"portfolio": portfolio})
