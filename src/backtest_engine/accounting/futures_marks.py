"""Carry existing futures exposure from its last valuation to the next fill."""

from dataclasses import dataclass
from typing import Any

from backtest_engine.accounting.futures import contract_point_value
from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime
from backtest_engine.core.exceptions import DatasetValidationError


@dataclass(frozen=True)
class FuturesMark:
    symbol: str
    price: float
    point_value: float
    currency: str


class FuturesMarkBook:
    def __init__(self) -> None:
        self.marks: dict[str, FuturesMark] = {}

    def advance(
        self,
        *,
        key: str,
        previous_quantity: float,
        fill_symbol: str,
        fill_price: float,
        exit_symbol: str,
        exit_price: float,
        exit_instrument: Any,
        ledger: BacktestLedgerRuntime,
        equity_fraction: float | None = None,
    ) -> float:
        previous = self.marks.get(key)
        gap = 0.0
        if previous is not None and previous_quantity:
            if previous.symbol != fill_symbol:
                raise DatasetValidationError(
                    f"{key}: carried futures position crosses {previous.symbol}/{fill_symbol} "
                    "before the execution interval; explicit roll fills are required"
                )
            if equity_fraction is not None:
                gap = equity_fraction * (fill_price / previous.price - 1.0)
            else:
                gap = ledger.amount_in_base(
                    previous_quantity * (fill_price - previous.price) * previous.point_value,
                    previous.currency,
                )
        self.marks[key] = FuturesMark(
            exit_symbol,
            exit_price,
            contract_point_value(exit_instrument),
            exit_instrument.pnl_currency,
        )
        return gap
