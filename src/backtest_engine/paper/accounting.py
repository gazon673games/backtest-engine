"""Paper portfolio backed by the production ledger and perpetual account."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Any

from backtest_engine.accounting.contract_support import validate_accounting_contract
from backtest_engine.accounting.funding import funding_payment_for_row
from backtest_engine.accounting.funding_policy import FundingMode
from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime
from backtest_engine.accounting.perpetual import (
    LinearPerpetualAccount,
)
from backtest_engine.accounting.perpetual_inputs import (
    perpetual_margin_schedule,
    perpetual_mark_prices,
)
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.instrument_registry import InstrumentSpec
from backtest_engine.execution.models import FillEvent


class PaperAccounting:
    def __init__(
        self,
        *,
        instrument: InstrumentSpec,
        initial_capital: float,
        base_currency: str,
        collateral_currency: str | None,
        leverage: float,
        liquidity_role: str,
        require_mark_price: bool = True,
        require_margin_tiers: bool = True,
        funding_mode: FundingMode = "legacy",
    ) -> None:
        if instrument.instrument_type not in {"spot", "perpetual"}:
            raise DatasetValidationError("Paper accounting supports spot and perpetual instruments")
        validate_accounting_contract(instrument)
        self.instrument = instrument
        self.initial_capital = initial_capital
        self.collateral_currency = collateral_currency or instrument.settlement_currency
        self.leverage = leverage
        self.liquidity_role = liquidity_role
        self.require_mark_price = require_mark_price
        self.require_margin_tiers = require_margin_tiers
        self.funding_mode = funding_mode
        self.ledger = BacktestLedgerRuntime(
            initial_capital=initial_capital,
            base_currency=base_currency,
        )
        self.perpetual: LinearPerpetualAccount | None = None
        self.started = False
        self.equity_cash = initial_capital

    def settle_event(self, row: dict[str, Any], fills: list[FillEvent]) -> dict[str, Any]:
        ts = as_datetime(row["ts"])
        self._start(ts)
        previous_equity = self.equity_cash
        if self.instrument.instrument_type == "spot":
            costs = self._settle_spot(ts, row, fills)
            liquidation = None
        else:
            costs, liquidation = self._settle_perpetual(ts, row, fills)
        self._append_snapshot(ts, previous_equity, costs)
        return {
            "equity_cash": self.equity_cash,
            "cost_cash": self.ledger.amount_in_base(costs, self._settlement_currency()),
            "liquidated": liquidation is not None and liquidation.liquidated,
            "liquidation_status": None if liquidation is None else liquidation.status,
        }

    def _settle_spot(self, ts: datetime, row: dict[str, Any], fills: list[FillEvent]) -> float:
        base = self.instrument.base_currency
        quote = self.instrument.quote_currency
        if not base or not quote:
            raise DatasetValidationError("Spot paper accounting requires base and quote currencies")
        total_fee = 0.0
        point = self.instrument.contract_multiplier * self.instrument.lot_size
        for fill in fills:
            quantity = fill.quantity * point
            fee = quantity * fill.price * self._fee_bps(fill) / 10_000.0
            self.ledger.ledger.apply_spot_trade(
                ts=ts,
                symbol=self.instrument.symbol,
                side=fill.side,
                quantity=quantity,
                price=fill.price,
                base_currency=base,
                quote_currency=quote,
                fee_amount=fee,
                fee_currency=quote,
            )
            total_fee += fee
        mark = market_price(row)
        self.ledger.update_market_rate(base, quote, mark)
        self.equity_cash = self.ledger.equity_cash
        return total_fee

    def _settle_perpetual(self, ts: datetime, row: dict[str, Any], fills: list[FillEvent]) -> Any:
        currency = self._settlement_currency()
        account = self._perpetual_account(currency)
        account.wallet_balance = self.ledger.ledger.cash.get(currency, 0.0)
        total_cost = 0.0
        for fill in fills:
            delta = fill.quantity if fill.side == "buy" else -fill.quantity
            result = account.apply_fill(delta, fill.price, self._fee_bps(fill))
            self.ledger.ledger.apply_variation_margin(
                ts=ts,
                symbol=self.instrument.symbol,
                amount=result.realized_pnl,
                currency=currency,
            )
            if result.fee:
                self.ledger.ledger.apply_fee(
                    ts=ts,
                    symbol=self.instrument.symbol,
                    amount=result.fee,
                    currency=currency,
                )
            total_cost += result.fee
        mark, mark_low, mark_high = perpetual_mark_prices(
            row, market_price(row), require_mark_price=self.require_mark_price
        )
        funding = account.apply_funding_payment(
            funding_payment_for_row(
                row,
                quantity=account.quantity,
                point_value=account.point_value,
                mark_price=mark,
                rate_column="funding_rate",
                default_rate=0.0,
                event_column="funding_event",
                require_event_flag=True,
                mode=self.funding_mode,
            ),
        )
        self._post_signed_cost(ts, currency, funding)
        total_cost += funding
        liquidation = account.liquidate_if_crossed(
            mark_low=mark_low,
            mark_high=mark_high,
            quantity_step=self.instrument.quantity_step,
        )
        if liquidation.liquidated:
            self.ledger.ledger.apply_variation_margin(
                ts=ts,
                symbol=self.instrument.symbol,
                amount=liquidation.realized_pnl,
                currency=currency,
            )
            self._post_signed_cost(ts, currency, liquidation.fee)
            total_cost += liquidation.fee
        self.equity_cash = self.ledger.equity_cash + self.ledger.amount_in_base(
            account.unrealized_pnl(mark), currency
        )
        return total_cost, liquidation

    def _perpetual_account(self, currency: str) -> LinearPerpetualAccount:
        if self.perpetual is not None:
            return self.perpetual
        self.perpetual = LinearPerpetualAccount(
            wallet_balance=self.ledger.ledger.cash.get(currency, 0.0),
            point_value=self.instrument.contract_multiplier * self.instrument.lot_size,
            leverage=self.leverage,
            margin_schedule=perpetual_margin_schedule(
                self.instrument, require_margin_tiers=self.require_margin_tiers
            ),
            liquidation_fee_bps=self.instrument.liquidation_fee_bps,
        )
        return self.perpetual

    def _start(self, ts: datetime) -> None:
        if self.started:
            return
        currency = self.collateral_currency or self.ledger.base_currency
        self.ledger.ledger.deposit(
            ts=ts, currency=currency, amount=self.initial_capital, reason="paper_initial_capital"
        )
        self.started = True

    def _post_signed_cost(self, ts: datetime, currency: str, amount: float) -> None:
        if amount >= 0.0:
            self.ledger.ledger.apply_fee(
                ts=ts, symbol=self.instrument.symbol, amount=amount, currency=currency
            )
        else:
            self.ledger.ledger.apply_variation_margin(
                ts=ts, symbol=self.instrument.symbol, amount=-amount, currency=currency
            )

    def _fee_bps(self, fill: FillEvent) -> float:
        role = (
            fill.liquidity_role
            if fill.liquidity_role in {"maker", "taker"}
            else self.liquidity_role
        )
        return self.instrument.maker_fee_bps if role == "maker" else self.instrument.taker_fee_bps

    def _settlement_currency(self) -> str:
        currency = self.instrument.settlement_currency or self.instrument.quote_currency
        if not currency:
            raise DatasetValidationError("Paper instrument requires settlement currency")
        return currency

    def _append_snapshot(self, ts: datetime, previous: float, costs: float) -> None:
        self.ledger.snapshots.append(
            {
                "ts": ts,
                "symbol": self.instrument.symbol,
                "currency": self._settlement_currency(),
                "gross_pnl": self.equity_cash - previous + costs,
                "cost_cash": costs,
                "net_cash_flow": self.equity_cash - previous,
                "cash_balance": self.ledger.ledger.cash.get(self._settlement_currency(), 0.0),
                "equity_cash": self.equity_cash,
                "reported_equity_cash": self.equity_cash,
                "reconciliation_diff": 0.0,
            }
        )


def market_price(row: dict[str, Any]) -> float:
    value = row.get("mark_price") or row.get("close") or row.get("trade_price")
    if value is None and row.get("bid") is not None and row.get("ask") is not None:
        value = (float(row["bid"]) + float(row["ask"])) / 2.0
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0:
        raise DatasetValidationError("Paper mark price must be positive")
    return parsed


def as_datetime(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise DatasetValidationError(f"Paper event ts must be datetime, got {value!r}")
    return value
