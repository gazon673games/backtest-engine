"""Value objects shared by the multi-currency ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CashFlow:
    ts: datetime
    currency: str
    amount: float
    reason: str
    symbol: str | None = None


@dataclass(frozen=True)
class PositionBalance:
    symbol: str
    currency: str
    quantity: float = 0.0
    avg_price: float = 0.0
    contract_multiplier: float = 1.0


@dataclass(frozen=True)
class CommissionSchedule:
    currency: str
    fixed_per_order: float = 0.0
    bps: float = 0.0
    per_contract: float = 0.0


@dataclass(frozen=True)
class ReconciliationBreak:
    key: str
    expected: float
    actual: float
    diff: float
