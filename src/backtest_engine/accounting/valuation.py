"""Causal decisions consume the native ledger journal, including pending flows."""

import heapq
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

from backtest_engine.accounting.historical_rates import HistoricalRateBook
from backtest_engine.accounting.ledger import MultiCurrencyLedger, fx_rate
from backtest_engine.accounting.ledger_models import CashFlow
from backtest_engine.core.exceptions import DatasetValidationError


@dataclass(frozen=True)
class MarketFixing:
    ts: datetime
    source: str
    target: str
    rate: float


@dataclass(frozen=True)
class PositionFixing:
    ts: datetime
    symbol: str
    currency: str
    unrealized: float


@dataclass(frozen=True)
class DecisionBalances:
    cash: dict[str, float]
    markets: dict[tuple[str, str], float]
    positions: dict[str, tuple[str, float]]


class CausalLedgerView:
    """Advance each recorded flow once, retaining only outstanding future events."""

    def __init__(self, ledger: MultiCurrencyLedger, history_size: int = 4) -> None:
        self.ledger = ledger
        self.cursor = 0
        self.sequence = 0
        self.pending: list[tuple[datetime, int, CashFlow | MarketFixing | PositionFixing]] = []
        self.cash: dict[str, float] = {}
        self.markets: dict[tuple[str, str], float] = {}
        self.positions: dict[str, tuple[str, float]] = {}
        self.current_ts: datetime | None = None
        self.history_size = history_size
        self.history: OrderedDict[datetime, DecisionBalances] = OrderedDict()

    def observe(self, event: MarketFixing | PositionFixing | CashFlow) -> None:
        heapq.heappush(self.pending, (event.ts, self.sequence, event))
        self.sequence += 1

    def balances_at(self, ts: datetime) -> DecisionBalances:
        if self.current_ts is not None and ts < self.current_ts:
            if ts in self.history:
                return self.history[ts]
            raise DatasetValidationError(f"Decision at {ts} predates retained ledger history")
        if (
            ts == self.current_ts
            and self.cursor == len(self.ledger.cash_flows)
            and (not self.pending or self.pending[0][0] > ts)
        ):
            return self.history[ts]
        for flow in self.ledger.cash_flows[self.cursor :]:
            self.observe(flow)
        self.cursor = len(self.ledger.cash_flows)
        while self.pending and self.pending[0][0] <= ts:
            event = heapq.heappop(self.pending)[2]
            if isinstance(event, CashFlow):
                self.cash[event.currency] = self.cash.get(event.currency, 0.0) + event.amount
            elif isinstance(event, MarketFixing):
                self.markets[(event.source, event.target)] = event.rate
            else:
                self.positions[event.symbol] = (event.currency, event.unrealized)
        self.current_ts = ts
        point = DecisionBalances(dict(self.cash), dict(self.markets), dict(self.positions))
        self.history[ts] = point
        while len(self.history) > self.history_size:
            self.history.popitem(last=False)
        return point

    def equity_at(
        self, ts: datetime, static_rates: dict[str, float], book: HistoricalRateBook | None
    ) -> float:
        balances = self.balances_at(ts)
        raw_rates = book.rates_at(ts, static_rates) if book is not None else static_rates
        rates = {tuple(pair.split("/")): rate for pair, rate in raw_rates.items()}
        rates.update(balances.markets)
        base = self.ledger.base_currency
        cash = sum(
            amount * fx_rate(currency, base, rates)
            for currency, amount in balances.cash.items()
            if amount
        )
        return cash + sum(
            amount * fx_rate(currency, base, rates)
            for currency, amount in balances.positions.values()
            if amount
        )
