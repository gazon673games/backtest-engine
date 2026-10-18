from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time


@dataclass(frozen=True)
class CallSchedule:
    calls: dict[int, time | None]
    missing_closes: dict[int, list[time]]
    skipped_dates: set[date]
    every_bar: bool = False

    def should_call(self, index: int) -> bool:
        return self.every_bar or index in self.calls

    def call_time(self, index: int) -> time | None:
        return self.calls.get(index)


@dataclass
class ActiveTrade:
    trade_id: int
    position: float
    entry_contracts: float
    current_contracts: float
    entry_ts: datetime
    entry_price: float
    entry_notional: float
    entry_active_contract: str
    entry_source_contract_symbol: str
    entry_equity_cash: float
    cost_cash: float
    gross_pnl_cash: float
    entry_point_value: float
    current_point_value: float
    holding_bars: int
    price_path: list[float]
