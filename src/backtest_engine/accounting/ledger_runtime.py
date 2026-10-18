"""Makes the multi-currency ledger the accounting source of truth during a run."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from backtest_engine.accounting.fx import currency_code
from backtest_engine.accounting.historical_rates import HistoricalRateBook
from backtest_engine.accounting.ledger import MultiCurrencyLedger, fx_rate
from backtest_engine.accounting.valuation import CausalLedgerView, MarketFixing, PositionFixing
from backtest_engine.core.exceptions import DatasetValidationError


class BacktestLedgerRuntime:
    def __init__(
        self,
        *,
        initial_capital: float,
        base_currency: str,
        fx_rates: dict[str, float] | None = None,
        start_ts: datetime | None = None,
        collateral_currency: str | None = None,
        initial_collateral: float | None = None,
        rate_book: HistoricalRateBook | None = None,
        history_size: int = 3,
    ) -> None:
        self.base_currency = base_currency
        self.fx_rates = parse_fx_rates(fx_rates or {})
        self.rate_book = rate_book
        self.static_rates = dict(fx_rates or {})
        self.ledger = MultiCurrencyLedger(base_currency=base_currency)
        self.snapshots: list[dict[str, Any]] = []
        self.decisions = CausalLedgerView(self.ledger, history_size)
        if start_ts is not None:
            self.ledger.deposit(
                ts=start_ts,
                currency=(collateral_currency if initial_collateral is not None else base_currency),
                amount=initial_collateral or initial_capital,
                reason="initial_capital",
            )
            if initial_collateral is not None and not math.isclose(
                self.equity_cash, initial_capital, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise DatasetValidationError(
                    "initial_collateral value does not match initial_capital: "
                    f"valued={self.equity_cash} {base_currency}, configured={initial_capital}"
                )

    @property
    def equity_cash(self) -> float:
        return self.ledger.equity(self.fx_rates)

    def record_unrealized(self, ts: datetime, symbol: str, currency: str, amount: float) -> None:
        self.decisions.observe(PositionFixing(ts, symbol, currency, amount))

    def decision_equity_at(self, ts: datetime) -> float:
        return self.decisions.equity_at(ts, self.static_rates, self.rate_book)

    def decision_cash_at(self, ts: datetime) -> dict[str, float]:
        return dict(self.decisions.balances_at(ts).cash)

    def amount_in_base(self, amount: float, currency: str) -> float:
        return amount * fx_rate(
            currency_code(currency, "currency"),
            currency_code(self.base_currency, "base_currency"),
            self.fx_rates,
        )

    def update_fx_rates(self, raw_rates: dict[str, float]) -> None:
        self.fx_rates.update(parse_fx_rates(raw_rates))
        if self.rate_book is None:
            self.static_rates.update(raw_rates)

    def activate_rates(self, ts: datetime) -> dict[str, float]:
        rates = (
            self.rate_book.rates_at(ts, self.static_rates)
            if self.rate_book is not None
            else self.static_rates
        )
        self.update_fx_rates(rates)
        return rates

    def update_market_rate(
        self, source: str, target: str, rate: float, ts: datetime | None = None
    ) -> None:
        if not math.isfinite(rate) or rate <= 0.0:
            raise DatasetValidationError("Market valuation rate must be finite and positive")
        self.fx_rates[(source.strip().upper(), target.strip().upper())] = rate
        if ts is not None:
            self.decisions.observe(
                MarketFixing(ts, source.strip().upper(), target.strip().upper(), rate)
            )

    def apply_spot_interval(
        self,
        *,
        fill_ts: datetime,
        valuation_ts: datetime,
        symbol: str,
        base_currency: str,
        quote_currency: str,
        quantity_delta: float,
        fill_price: float,
        mark_price: float,
        trading_fee_quote: float,
        additional_cost_quote: float,
        previous_equity_cash: float,
    ) -> float:
        self.activate_rates(fill_ts)
        fee_base = self.amount_in_base(trading_fee_quote + additional_cost_quote, quote_currency)
        if quantity_delta:
            self.ledger.apply_spot_trade(
                ts=fill_ts,
                symbol=symbol,
                side="buy" if quantity_delta > 0.0 else "sell",
                quantity=abs(quantity_delta),
                price=fill_price,
                base_currency=base_currency,
                quote_currency=quote_currency,
                fee_amount=trading_fee_quote,
                fee_currency=quote_currency,
            )
        if additional_cost_quote:
            self.ledger.apply_fee(
                ts=fill_ts,
                symbol=symbol,
                amount=additional_cost_quote,
                currency=quote_currency,
            )
        self.update_market_rate(base_currency, quote_currency, fill_price, fill_ts)
        self.activate_rates(valuation_ts)
        self.update_market_rate(base_currency, quote_currency, mark_price, valuation_ts)
        equity_cash = self.equity_cash
        self.snapshots.append(
            {
                "ts": valuation_ts,
                "symbol": symbol,
                "currency": quote_currency,
                "gross_pnl": equity_cash - previous_equity_cash + fee_base,
                "cost_cash": fee_base,
                "net_cash_flow": equity_cash - previous_equity_cash,
                "cash_balance": self.ledger.cash.get(quote_currency, 0.0),
                "equity_cash": equity_cash,
                "reported_equity_cash": equity_cash,
                "reconciliation_diff": 0.0,
            }
        )
        return equity_cash

    def apply_event_cash_flow(
        self,
        *,
        ts: datetime,
        symbol: str,
        currency: str,
        gross_pnl: float,
        cost_cash: float,
        fill_ts: datetime | None = None,
        opening_pnl: float = 0.0,
        previous_equity_cash: float | None = None,
    ) -> float:
        previous_equity = self.equity_cash if previous_equity_cash is None else previous_equity_cash
        self.activate_rates(fill_ts or ts)
        fee_base = self.amount_in_base(cost_cash, currency)
        if opening_pnl:
            self.ledger.apply_variation_margin(
                ts=fill_ts or ts, symbol=symbol, amount=opening_pnl, currency=currency
            )
        if cost_cash >= 0.0:
            self.ledger.apply_fee(
                ts=fill_ts or ts,
                symbol=symbol,
                amount=cost_cash,
                currency=currency,
            )
        else:
            self.ledger.apply_variation_margin(
                ts=fill_ts or ts,
                symbol=symbol,
                amount=-cost_cash,
                currency=currency,
            )
        self.activate_rates(ts)
        self.ledger.apply_variation_margin(
            ts=ts,
            symbol=symbol,
            amount=gross_pnl - opening_pnl,
            currency=currency,
        )
        equity_cash = self.equity_cash
        self.snapshots.append(
            {
                "ts": ts,
                "symbol": symbol,
                "currency": currency,
                "gross_pnl": equity_cash - previous_equity + fee_base,
                "cost_cash": fee_base,
                "net_cash_flow": equity_cash - previous_equity,
                "cash_balance": self.ledger.cash.get(currency, 0.0),
                "equity_cash": equity_cash,
                "reported_equity_cash": equity_cash,
                "reconciliation_diff": 0.0,
            }
        )
        return equity_cash


def parse_fx_rates(raw: dict[str, float]) -> dict[tuple[str, str], float]:
    rates: dict[tuple[str, str], float] = {}
    for key, value in raw.items():
        if "/" not in key:
            raise DatasetValidationError(f"Invalid FX rate key: {key}. Expected SOURCE/TARGET")
        source, target = key.split("/", 1)
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise DatasetValidationError(f"FX rate must be finite and positive: {key}={value}")
        rates[(source.strip().upper(), target.strip().upper())] = parsed
    return rates


def amount_in_base_currency(
    amount: float,
    currency: str,
    base_currency: str,
    raw_fx_rates: dict[str, float],
) -> float:
    source = currency_code(currency, "currency")
    target = currency_code(base_currency, "base_currency")
    if source == target:
        return amount
    return amount * fx_rate(
        source,
        target,
        parse_fx_rates(raw_fx_rates),
    )


def amount_from_base_currency(
    amount: float,
    currency: str,
    base_currency: str,
    raw_fx_rates: dict[str, float],
) -> float:
    source = currency_code(currency, "currency")
    target = currency_code(base_currency, "base_currency")
    if source == target:
        return amount
    rate = fx_rate(
        source,
        target,
        parse_fx_rates(raw_fx_rates),
    )
    return amount / rate


def symbol_from_row(row: dict[str, Any], default: str = "") -> str:
    return str(row.get("active_contract") or row.get("asset_id") or default)
