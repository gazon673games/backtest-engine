"""Bridges generic event rows to crypto accounting primitives."""

from __future__ import annotations

from typing import Any

from backtest_engine.accounting.crypto import CryptoSettlement, settle_crypto_interval
from backtest_engine.accounting.fees import trading_fee_bps
from backtest_engine.accounting.funding import funding_rate_for_row
from backtest_engine.intraday.event.config import (
    IntradayEventExecutionConfig,
    IntradayEventPortfolioConfig,
)
from backtest_engine.intraday.event.futures import native_cash_in_base
from backtest_engine.intraday.event.utils import ratio_float


def apply_crypto_costs(
    *,
    exit_row: dict[str, Any],
    fill_price: float,
    exit_price: float,
    current_contracts: float,
    contract_delta: float,
    point_value_native: float,
    instrument: Any,
    execution: IntradayEventExecutionConfig,
    portfolio: IntradayEventPortfolioConfig,
    previous_equity_cash: float,
    cost_parts: dict[str, float],
) -> tuple[dict[str, float], CryptoSettlement | None]:
    if instrument.instrument_type not in {"spot", "perpetual"}:
        return with_crypto_defaults(cost_parts), None
    if instrument.instrument_type == "perpetual":
        # Stateful perpetual accounting is applied after the order/position transition.
        return with_crypto_defaults(cost_parts), None
    fee_bps = trading_fee_bps(instrument, execution.liquidity_role, execution.commission_bps)
    funding_rate = funding_rate_for_row(
        exit_row,
        rate_column=execution.funding_rate_column,
        default_rate=execution.default_funding_rate,
        event_column=execution.funding_event_column,
        require_event_flag=execution.require_funding_event_flag,
    )
    cash_to_base = native_cash_in_base(1.0, instrument, portfolio)
    equity_native = previous_equity_cash / cash_to_base
    if portfolio.margin_mode == "isolated" and current_contracts:
        equity_native = (
            abs(current_contracts) * fill_price * point_value_native / portfolio.leverage
        )
    settlement = settle_crypto_interval(
        instrument_type=instrument.instrument_type,
        contract_type=instrument.contract_type,
        quantity=current_contracts,
        quantity_delta=contract_delta,
        point_value=point_value_native,
        fill_price=fill_price,
        exit_price=exit_price,
        low_price=float(exit_row["low"]),
        high_price=float(exit_row["high"]),
        equity=equity_native,
        fee_bps=fee_bps,
        funding_rate=funding_rate,
        maintenance_margin_rate=instrument.maintenance_margin_rate,
        liquidation_fee_bps=instrument.liquidation_fee_bps,
    )
    trading_fee = native_cash_in_base(settlement.trading_fee, instrument, portfolio)
    funding_cost = native_cash_in_base(settlement.funding_payment, instrument, portfolio)
    liquidation_fee = native_cash_in_base(settlement.liquidation_fee, instrument, portfolio)
    updated = dict(cost_parts)
    # A declared maker/taker schedule replaces the generic commission.
    updated["commission_cost_cash"] = trading_fee
    updated["trading_fee_cash"] = trading_fee
    updated["funding_cost_cash"] = funding_cost
    updated["liquidation_fee_cash"] = liquidation_fee
    updated["total_cost_cash"] = (
        trading_fee
        + updated["slippage_cost_cash"]
        + updated["spread_cost_cash"]
        + updated["roll_cost_cash"]
        + funding_cost
        + liquidation_fee
    )
    updated["total_cost_return"] = ratio_float(updated["total_cost_cash"], previous_equity_cash)
    return updated, settlement


def with_crypto_defaults(cost_parts: dict[str, float]) -> dict[str, float]:
    result = dict(cost_parts)
    result.setdefault("trading_fee_cash", 0.0)
    result.setdefault("funding_cost_cash", 0.0)
    result.setdefault("liquidation_fee_cash", 0.0)
    return result
