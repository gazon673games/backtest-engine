"""Connects stateful perpetual accounts to the event ledger."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from backtest_engine.accounting.crypto import CryptoSettlement
from backtest_engine.accounting.fees import trading_fee_bps
from backtest_engine.accounting.funding import funding_payment_for_row
from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime, symbol_from_row
from backtest_engine.accounting.perpetual import (
    LinearPerpetualAccount,
)
from backtest_engine.accounting.perpetual_inputs import (
    market_value as market_value,
)
from backtest_engine.accounting.perpetual_inputs import (
    perpetual_margin_schedule,
    perpetual_mark_prices,
)
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig


class PerpetualEventAccounting:
    def __init__(self, ledger: BacktestLedgerRuntime) -> None:
        self.ledger = ledger
        self.accounts: dict[str, LinearPerpetualAccount] = {}
        self.specifications: dict[str, Any] = {}

    def settle(self, **values: Any) -> dict[str, Any]:
        instrument = values["instrument"]
        config = values["config"]
        row = values["exit_row"]
        symbol = symbol_from_row(values["fill_row"])
        currency = instrument.settlement_currency or instrument.pnl_currency
        self.ledger.activate_rates(values["fill_ts"])
        fill_rate = self.ledger.amount_in_base(1.0, currency)
        account = self.accounts.get(symbol)
        if account is None or self.specifications.get(symbol) is not instrument:
            updated = build_perpetual_account(
                instrument,
                config,
                self.ledger.ledger.cash.get(currency, 0.0)
                if account is None
                else account.wallet_balance,
            )
            if account is not None:
                updated.quantity, updated.entry_price = account.quantity, account.entry_price
            account = updated
            self.accounts[symbol] = account
            self.specifications[symbol] = instrument
        if config.portfolio.margin_mode == "cross":
            # The ledger owns collateral. Re-sync before each transition so other
            # realized cash flows cannot leave the margin wallet stale.
            account.wallet_balance = self.ledger.ledger.cash.get(currency, 0.0)
        mark, mark_low, mark_high = mark_prices(row, values["mark_price"], config)
        fee_bps = trading_fee_bps(
            instrument, config.execution.liquidity_role, config.execution.commission_bps
        )
        previous_account = (account.wallet_balance, account.quantity, account.entry_price)
        fill = account.apply_fill(values["quantity_delta"], values["fill_price"], fee_bps)
        generic_native = base_to_native_generic_costs(values["cost_parts"], self.ledger, currency)
        account.wallet_balance -= generic_native
        if abs(account.quantity) > abs(previous_account[1]) and account.initial_margin(
            values["fill_price"]
        ) > account.equity(values["fill_price"]):
            account.wallet_balance, account.quantity, account.entry_price = previous_account
            raise DatasetValidationError(
                f"Perpetual order exceeds initial margin at leverage={config.portfolio.leverage}"
            )
        self._sync_native_cash(values["fill_ts"], symbol, currency, fill.realized_pnl, fill.fee)
        if generic_native:
            self.ledger.ledger.apply_fee(
                ts=values["fill_ts"], symbol=symbol, amount=generic_native, currency=currency
            )
        self.ledger.record_unrealized(
            values["fill_ts"], symbol, currency, account.unrealized_pnl(values["fill_price"])
        )
        self.ledger.activate_rates(values["valuation_ts"])
        funding_payment = funding_payment_for_row(
            row,
            quantity=account.quantity,
            point_value=account.point_value,
            mark_price=mark,
            rate_column=config.execution.funding_rate_column,
            default_rate=config.execution.default_funding_rate,
            event_column=config.execution.funding_event_column,
            require_event_flag=config.execution.require_funding_event_flag,
            mode=config.execution.funding_mode,
        )
        funding = account.apply_funding_payment(funding_payment)
        self._sync_cost(values["valuation_ts"], symbol, currency, funding)
        liquidation = account.liquidate_if_crossed(
            mark_low=mark_low,
            mark_high=mark_high,
            quantity_step=instrument.quantity_step,
            target_buffer=config.portfolio.liquidation_target_buffer,
            minimum_close_fraction=config.portfolio.minimum_liquidation_fraction,
        )
        if liquidation.liquidated:
            self._sync_native_cash(
                values["valuation_ts"],
                symbol,
                currency,
                liquidation.realized_pnl,
                liquidation.fee,
            )
        effective_mark = liquidation.price if liquidation.liquidated else mark
        equity = self._portfolio_equity(effective_mark, symbol, currency)
        self.ledger.record_unrealized(
            values["valuation_ts"], symbol, currency, account.unrealized_pnl(effective_mark)
        )
        update_perpetual_costs(
            values["cost_parts"],
            self.ledger,
            currency,
            fill.fee,
            funding,
            liquidation.fee,
            fill_rate,
        )
        update_pnl_state(
            values["pnl_state"],
            equity,
            values["previous_equity_cash"],
            values["cost_parts"]["total_cost_cash"],
        )
        self._append_snapshot(
            values["valuation_ts"],
            symbol,
            currency,
            equity,
            values["previous_equity_cash"],
            values["pnl_state"],
        )
        settlement = CryptoSettlement(
            trading_fee=fill.fee,
            funding_payment=funding,
            maintenance_margin=account.maintenance_margin(effective_mark),
            liquidation_fee=liquidation.fee,
            liquidation_price=liquidation.price,
            liquidated=liquidation.liquidated,
            effective_exit_price=effective_mark,
            bankruptcy_price=liquidation.bankruptcy_price,
            liquidation_quantity=liquidation.closed_quantity,
            remaining_quantity=account.quantity,
            liquidation_status=liquidation.status,
        )
        return {
            "equity_cash": equity,
            "crypto_settlement": settlement,
            "ending_quantity": account.quantity,
            "effective_exit_price": effective_mark,
        }

    def _sync_native_cash(
        self,
        ts: datetime,
        symbol: str,
        currency: str,
        realized_pnl: float,
        fee: float,
    ) -> None:
        self.ledger.ledger.apply_variation_margin(
            ts=ts, symbol=symbol, amount=realized_pnl, currency=currency
        )
        if fee:
            self.ledger.ledger.apply_fee(ts=ts, symbol=symbol, amount=fee, currency=currency)

    def _sync_cost(self, ts: datetime, symbol: str, currency: str, cost: float) -> None:
        if cost >= 0.0:
            self.ledger.ledger.apply_fee(ts=ts, symbol=symbol, amount=cost, currency=currency)
        else:
            self.ledger.ledger.apply_variation_margin(
                ts=ts, symbol=symbol, amount=-cost, currency=currency
            )

    def _portfolio_equity(self, mark: float, symbol: str, currency: str) -> float:
        unrealized = self.accounts[symbol].unrealized_pnl(mark)
        return self.ledger.equity_cash + self.ledger.amount_in_base(unrealized, currency)

    def _append_snapshot(
        self,
        ts: datetime,
        symbol: str,
        currency: str,
        equity: float,
        previous_equity: float,
        pnl: dict[str, float],
    ) -> None:
        self.ledger.snapshots.append(
            {
                "ts": ts,
                "symbol": symbol,
                "currency": currency,
                "gross_pnl": pnl["gross_pnl"],
                "cost_cash": pnl["cost_cash"],
                "net_cash_flow": equity - previous_equity,
                "cash_balance": self.ledger.ledger.cash.get(currency, 0.0),
                "equity_cash": equity,
                "reported_equity_cash": equity,
                "reconciliation_diff": 0.0,
            }
        )


def build_perpetual_account(
    instrument: Any,
    config: IntradayEventBacktestConfig,
    wallet_balance: float,
) -> LinearPerpetualAccount:
    return LinearPerpetualAccount(
        wallet_balance=wallet_balance,
        point_value=instrument.contract_multiplier * instrument.lot_size,
        leverage=config.portfolio.leverage,
        margin_schedule=perpetual_margin_schedule(
            instrument, require_margin_tiers=config.portfolio.require_margin_tiers
        ),
        liquidation_fee_bps=instrument.liquidation_fee_bps,
    )


def mark_prices(
    row: dict[str, Any], fallback: float, config: IntradayEventBacktestConfig
) -> tuple[float, float, float]:
    return perpetual_mark_prices(
        row,
        fallback,
        require_mark_price=config.portfolio.require_mark_price,
        mark_price_column=config.execution.mark_price_column,
        mark_low_column=config.execution.mark_low_column,
        mark_high_column=config.execution.mark_high_column,
    )


def base_to_native_generic_costs(
    costs: dict[str, float], ledger: BacktestLedgerRuntime, currency: str
) -> float:
    base_cost = sum(
        costs[key] for key in ("slippage_cost_cash", "spread_cost_cash", "roll_cost_cash")
    )
    return base_cost / ledger.amount_in_base(1.0, currency)


def update_perpetual_costs(
    costs: dict[str, float],
    ledger: BacktestLedgerRuntime,
    currency: str,
    trading_fee: float,
    funding: float,
    liquidation_fee: float,
    fill_rate: float,
) -> None:
    convert = ledger.amount_in_base(1.0, currency)
    costs["commission_cost_cash"] = trading_fee * fill_rate
    costs["trading_fee_cash"] = trading_fee * fill_rate
    costs["funding_cost_cash"] = funding * convert
    costs["liquidation_fee_cash"] = liquidation_fee * convert
    costs["total_cost_cash"] = sum(
        costs[key]
        for key in (
            "trading_fee_cash",
            "slippage_cost_cash",
            "spread_cost_cash",
            "roll_cost_cash",
            "funding_cost_cash",
            "liquidation_fee_cash",
        )
    )
    costs["total_cost_return"] = 0.0


def update_pnl_state(
    pnl: dict[str, float], equity: float, previous_equity: float, costs: float
) -> None:
    pnl["net_pnl"] = equity - previous_equity
    pnl["cost_cash"] = costs
    pnl["cost"] = costs / previous_equity
    pnl["gross_pnl"] = pnl["net_pnl"] + costs
    pnl["net_return"] = pnl["net_pnl"] / previous_equity
    pnl["gross_return"] = pnl["gross_pnl"] / previous_equity
