"""Sparse research valuations using the production ledger and perpetual account."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from backtest_engine.accounting.contract_support import validate_accounting_contract
from backtest_engine.accounting.funding import funding_payment_for_row
from backtest_engine.accounting.futures import contract_point_value
from backtest_engine.accounting.ledger import MultiCurrencyLedger
from backtest_engine.accounting.perpetual import LinearPerpetualAccount
from backtest_engine.accounting.perpetual_inputs import perpetual_margin_schedule
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.instrument_registry import InstrumentSpec


@dataclass(frozen=True)
class MarketSchedulePrices:
    opened: list
    available: list
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    marks: np.ndarray
    mark_lows: np.ndarray
    mark_highs: np.ndarray
    funding_unit: np.ndarray


def prices_from_frame(frame, *, perpetual=False):
    marks = frame["mark_price"].to_numpy() if perpetual else frame["close"].to_numpy()
    lows = frame["mark_low"].to_numpy() if perpetual else frame["low"].to_numpy()
    highs = frame["mark_high"].to_numpy() if perpetual else frame["high"].to_numpy()
    if any(not np.all(np.isfinite(values) & (values > 0)) for values in (marks, lows, highs)):
        raise DatasetValidationError("Candidate schedule requires valid mark prices on every bar")
    funding = np.zeros(frame.height)
    if perpetual and "funding_event" in frame.columns:
        rates = frame["funding_rate"].to_numpy()
        priced = (
            frame["funding_price_rate"].to_list()
            if "funding_price_rate" in frame.columns
            else [None] * frame.height
        )
        for index in np.flatnonzero(frame["funding_event"].to_numpy()):
            funding[index] = funding_payment_for_row(
                {
                    "funding_event": True,
                    "funding_rate": rates[index],
                    "funding_price_rate": priced[index],
                },
                quantity=1,
                point_value=1,
                mark_price=marks[index],
                rate_column="funding_rate",
                default_rate=0,
                event_column="funding_event",
                require_event_flag=True,
            )
    return MarketSchedulePrices(
        frame["ts"].to_list(),
        frame["available_at"].to_list(),
        *[frame[c].to_numpy() for c in ("open", "high", "low", "close", "volume")],
        marks,
        lows,
        highs,
        funding,
    )


class ScheduleAccount:
    def __init__(self, prices, spec: InstrumentSpec, capital: float):
        validate_accounting_contract(spec)
        self.prices, self.spec = prices, spec
        self.currency = spec.settlement_currency or spec.pnl_currency
        self.ledger = MultiCurrencyLedger(base_currency=self.currency)
        self.ledger.deposit(ts=prices.opened[0], currency=self.currency, amount=capital)
        self.perpetual = None
        if spec.instrument_type == "perpetual":
            self.perpetual = LinearPerpetualAccount(
                wallet_balance=capital,
                point_value=contract_point_value(spec),
                leverage=1,
                margin_schedule=perpetual_margin_schedule(spec, require_margin_tiers=False),
                liquidation_fee_bps=spec.liquidation_fee_bps,
            )
        elif spec.instrument_type != "spot":
            raise DatasetValidationError("Candidate schedule supports spot and linear perpetuals")
        self.equity = np.full(len(prices.closes), capital, dtype=np.float64)
        self.quantities = np.zeros(len(prices.closes))
        self.cursor = 0
        self.liquidations = []
        self.funding_paid = 0.0

    @property
    def quantity(self):
        if self.perpetual is not None:
            return self.perpetual.quantity
        return self.ledger.cash.get(self.spec.base_currency, 0.0)

    def value(self, index):
        if self.perpetual is not None:
            return self.ledger.equity() + self.perpetual.unrealized_pnl(self.prices.marks[index])
        return self.ledger.equity(
            {(self.spec.base_currency, self.currency): self.prices.closes[index]}
        )

    def post_perpetual_fill(self, index, delta, price, fee_bps):
        account = self.perpetual
        before = (account.wallet_balance, account.quantity, account.entry_price)
        fill = account.apply_fill(delta, price, fee_bps)
        if abs(account.quantity) > abs(before[1]) and account.initial_margin(
            price
        ) > account.equity(price):
            account.wallet_balance, account.quantity, account.entry_price = before
            raise DatasetValidationError("Candidate order exceeds initial margin at 1x leverage")
        self._post(self.prices.opened[index], fill.realized_pnl, fill.fee)
        return fill.fee

    def funding(self, index):
        if self.perpetual is None or self.quantity == 0:
            return
        payment = self.perpetual.apply_funding_payment(
            self.quantity * self.perpetual.point_value * self.prices.funding_unit[index]
        )
        self.ledger.deposit(
            ts=self.prices.available[index],
            currency=self.currency,
            amount=-payment,
            reason="funding",
        )
        self.funding_paid += payment
        self.check_wallet()

    def _post(self, ts, realized, fee):
        if realized:
            self.ledger.apply_variation_margin(
                ts=ts, symbol=self.spec.symbol, amount=realized, currency=self.currency
            )
        self.ledger.apply_fee(ts=ts, symbol=self.spec.symbol, amount=fee, currency=self.currency)
        self.check_wallet()

    def check_wallet(self):
        if self.perpetual is None:
            return
        cash = self.ledger.equity()
        wallet = self.perpetual.wallet_balance
        tolerance = 1e-9 + 1e-12 * abs(wallet)
        if not (isfinite(cash) and isfinite(wallet) and abs(cash - wallet) <= tolerance):
            raise DatasetValidationError("Candidate perpetual wallet diverged from ledger cash")

    def _mark(self, start, stop):
        if start >= stop:
            return
        self.quantities[start:stop] = self.quantity
        if self.perpetual is not None:
            self.equity[start:stop] = self.ledger.equity() + self.perpetual.unrealized_pnl(
                self.prices.marks[start:stop]
            )
        else:
            self.equity[start:stop] = self.ledger.cash.get(self.currency, 0) + (
                self.quantity * self.prices.closes[start:stop]
            )

    def advance(self, stop):
        """Mark unchanged ledger balances in bulk; resolve each crossed margin boundary."""
        start = self.cursor
        while start < stop and self.perpetual is not None and self.quantity:
            critical = self.perpetual.liquidation_price()
            if critical is None:
                break
            extrema = self.prices.mark_lows if self.quantity > 0 else self.prices.mark_highs
            crossed = (
                extrema[start:stop] <= critical
                if self.quantity > 0
                else extrema[start:stop] >= critical
            )
            indices = np.flatnonzero(crossed)
            if not len(indices):
                break
            index = start + int(indices[0])
            self._mark(start, index)
            liquidation = self.perpetual.liquidate_if_crossed(
                mark_low=float(self.prices.mark_lows[index]),
                mark_high=float(self.prices.mark_highs[index]),
                quantity_step=self.spec.quantity_step,
            )
            self._post(self.prices.available[index], liquidation.realized_pnl, liquidation.fee)
            self.liquidations.append({"index": index, **vars(liquidation)})
            self._mark(index, index + 1)
            start = index + 1
        self._mark(start, stop)
        self.cursor = stop

    def reconcile(self):
        reconstructed = {}
        for flow in self.ledger.cash_flows:
            reconstructed[flow.currency] = reconstructed.get(flow.currency, 0) + flow.amount
        if self.ledger.reconcile_cash(reconstructed, tolerance=1e-8):
            raise DatasetValidationError("Candidate cash-flow reconciliation failed")
        error = abs(self.value(-1) - self.equity[-1])
        if error > 1e-8:
            raise DatasetValidationError("Candidate final NAV reconciliation failed")
        return error
