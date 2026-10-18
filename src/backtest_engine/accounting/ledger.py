from __future__ import annotations

import math
from datetime import datetime

from backtest_engine.accounting.fx import currency_code
from backtest_engine.accounting.fx import resolve_fx_rate as fx_rate
from backtest_engine.accounting.ledger_models import (
    CashFlow,
    CommissionSchedule,
    PositionBalance,
    ReconciliationBreak,
)
from backtest_engine.core.exceptions import DatasetValidationError

__all__ = ["CashFlow", "CommissionSchedule", "MultiCurrencyLedger", "PositionBalance"]


class MultiCurrencyLedger:
    def __init__(self, *, base_currency: str) -> None:
        self.base_currency = currency_code(base_currency, "base_currency")
        self.cash: dict[str, float] = {}
        self.positions: dict[str, PositionBalance] = {}
        self.cash_flows: list[CashFlow] = []

    def deposit(
        self,
        *,
        ts: datetime,
        currency: str,
        amount: float,
        reason: str = "deposit",
    ) -> None:
        self._add_cash(ts=ts, currency=currency, amount=amount, reason=reason)

    def apply_trade(
        self,
        *,
        ts: datetime,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        currency: str,
        contract_multiplier: float = 1.0,
        commission: CommissionSchedule | None = None,
    ) -> None:
        if side not in {"buy", "sell"}:
            raise DatasetValidationError(f"Unsupported trade side: {side}")
        signed_quantity = quantity if side == "buy" else -quantity
        notional = signed_quantity * price * contract_multiplier
        self._add_cash(
            ts=ts,
            currency=currency,
            amount=-notional,
            reason="trade",
            symbol=symbol,
        )
        self._update_position(symbol, signed_quantity, price, currency, contract_multiplier)
        fee = commission_amount(quantity, price, contract_multiplier, commission)
        if fee != 0.0 and commission is not None:
            self._add_cash(
                ts=ts,
                currency=commission.currency,
                amount=-fee,
                reason="commission",
                symbol=symbol,
            )

    def apply_spot_trade(
        self,
        *,
        ts: datetime,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        base_currency: str,
        quote_currency: str,
        fee_amount: float = 0.0,
        fee_currency: str | None = None,
        allow_borrow: bool = False,
        tolerance: float = 1e-12,
    ) -> None:
        """Book a spot fill as two asset transfers, never as variation margin."""
        if side not in {"buy", "sell"}:
            raise DatasetValidationError(f"Unsupported spot trade side: {side}")
        if quantity <= 0.0 or price <= 0.0:
            raise DatasetValidationError("Spot trade quantity and price must be positive")
        if fee_amount < 0.0:
            raise DatasetValidationError("Spot trade fee_amount must be non-negative")
        base = base_currency.strip().upper()
        quote = quote_currency.strip().upper()
        fee_asset = (fee_currency or quote).strip().upper()
        signed_quantity = quantity if side == "buy" else -quantity
        quote_delta = -signed_quantity * price
        deltas = {base: signed_quantity, quote: quote_delta}
        deltas[fee_asset] = deltas.get(fee_asset, 0.0) - fee_amount
        if not allow_borrow:
            for currency, delta in deltas.items():
                resulting = self.cash.get(currency, 0.0) + delta
                if resulting < -tolerance:
                    raise DatasetValidationError(
                        "Spot trade requires unavailable inventory: "
                        f"symbol={symbol}, currency={currency}, required={-delta}, "
                        f"available={self.cash.get(currency, 0.0)}"
                    )
        self._add_cash(
            ts=ts,
            currency=base,
            amount=signed_quantity,
            reason="spot_base_fill",
            symbol=symbol,
        )
        self._add_cash(
            ts=ts,
            currency=quote,
            amount=quote_delta,
            reason="spot_quote_fill",
            symbol=symbol,
        )
        if fee_amount:
            self._add_cash(
                ts=ts,
                currency=fee_asset,
                amount=-fee_amount,
                reason="spot_fee",
                symbol=symbol,
            )

    def apply_variation_margin(
        self,
        *,
        ts: datetime,
        symbol: str,
        amount: float,
        currency: str,
    ) -> None:
        if amount == 0.0:
            return
        self._add_cash(
            ts=ts,
            currency=currency,
            amount=amount,
            reason="variation_margin",
            symbol=symbol,
        )

    def apply_fee(
        self,
        *,
        ts: datetime,
        symbol: str | None,
        amount: float,
        currency: str,
    ) -> None:
        if amount == 0.0:
            return
        self._add_cash(
            ts=ts,
            currency=currency,
            amount=-abs(amount),
            reason="fee",
            symbol=symbol,
        )

    def equity(self, fx_rates: dict[tuple[str, str], float] | None = None) -> float:
        rates = fx_rates or {}
        return sum(
            amount * fx_rate(currency, self.base_currency, rates)
            for currency, amount in self.cash.items()
        )

    def reconcile_cash(
        self,
        expected_cash: dict[str, float],
        *,
        tolerance: float = 0.01,
    ) -> list[ReconciliationBreak]:
        breaks = []
        currencies = sorted(set(expected_cash) | set(self.cash))
        for currency in currencies:
            expected = expected_cash.get(currency, 0.0)
            actual = self.cash.get(currency, 0.0)
            diff = actual - expected
            if abs(diff) > tolerance:
                breaks.append(
                    ReconciliationBreak(
                        key=f"cash:{currency}",
                        expected=expected,
                        actual=actual,
                        diff=diff,
                    )
                )
        return breaks

    def _add_cash(
        self,
        *,
        ts: datetime,
        currency: str,
        amount: float,
        reason: str,
        symbol: str | None = None,
    ) -> None:
        if not math.isfinite(amount):
            raise DatasetValidationError(
                f"Cash flow amount must be finite: currency={currency}, amount={amount}"
            )
        self.cash[currency] = self.cash.get(currency, 0.0) + amount
        self.cash_flows.append(
            CashFlow(
                ts=ts,
                currency=currency,
                amount=amount,
                reason=reason,
                symbol=symbol,
            )
        )

    def _update_position(
        self,
        symbol: str,
        signed_quantity: float,
        price: float,
        currency: str,
        contract_multiplier: float,
    ) -> None:
        current = self.positions.get(
            symbol,
            PositionBalance(
                symbol=symbol,
                currency=currency,
                contract_multiplier=contract_multiplier,
            ),
        )
        new_quantity = current.quantity + signed_quantity
        if new_quantity == 0.0:
            self.positions[symbol] = PositionBalance(
                symbol=symbol,
                quantity=0.0,
                avg_price=0.0,
                currency=currency,
                contract_multiplier=contract_multiplier,
            )
            return
        if current.quantity == 0.0 or current.quantity * signed_quantity > 0.0:
            avg_price = (
                current.avg_price * abs(current.quantity) + price * abs(signed_quantity)
            ) / abs(new_quantity)
        else:
            avg_price = current.avg_price
        self.positions[symbol] = PositionBalance(
            symbol=symbol,
            quantity=new_quantity,
            avg_price=avg_price,
            currency=currency,
            contract_multiplier=contract_multiplier,
        )


def commission_amount(
    quantity: float,
    price: float,
    contract_multiplier: float,
    schedule: CommissionSchedule | None,
) -> float:
    if schedule is None:
        return 0.0
    notional = abs(quantity) * price * contract_multiplier
    return (
        schedule.fixed_per_order
        + abs(quantity) * schedule.per_contract
        + notional * schedule.bps / 10_000.0
    )
