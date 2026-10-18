from __future__ import annotations

from typing import Any

from backtest_engine.accounting.futures import (
    FuturesInstrumentSpec,
    FuturesPortfolioSpec,
)
from backtest_engine.accounting.futures import (
    cost_breakdown as futures_cost_breakdown,
)
from backtest_engine.accounting.futures import (
    effective_tick_value as futures_effective_tick_value,
)
from backtest_engine.accounting.ledger_runtime import (
    amount_from_base_currency,
    amount_in_base_currency,
)
from backtest_engine.intraday.event._futures_specs import (
    futures_execution_cost_spec,
)
from backtest_engine.intraday.event.config import (
    IntradayEventExecutionConfig,
    IntradayEventInstrumentSpec,
    IntradayEventPortfolioConfig,
)
from backtest_engine.intraday.event.utils import ratio_float


def contract_costs(
    *,
    row: dict[str, Any],
    next_row: dict[str, Any],
    current_contracts: float,
    contract_delta: float,
    turnover_notional: float,
    fill_price: float,
    exit_price: float,
    instrument: IntradayEventInstrumentSpec,
    portfolio: IntradayEventPortfolioConfig,
    execution: IntradayEventExecutionConfig,
    previous_equity_cash: float,
    point_value: float,
) -> dict[str, float]:
    roll_notional = 0.0
    roll_turnover = 0.0
    if str(row["active_contract"]) != str(next_row["active_contract"]) and current_contracts != 0.0:
        roll_notional = abs(current_contracts) * exit_price * point_value
        roll_turnover = 2.0 * abs(current_contracts)
    roll_turnover_notional = 0.0
    if roll_turnover:
        roll_turnover_notional = abs(current_contracts) * (fill_price + exit_price) * point_value
    costs = futures_cost_breakdown(
        contract_delta=abs(contract_delta) + roll_turnover,
        turnover_notional_cash=turnover_notional + roll_turnover_notional,
        current_contracts=current_contracts,
        roll_notional_cash=roll_notional,
        instrument=futures_instrument_spec(instrument),
        execution=futures_execution_cost_spec(execution),
    )
    return cost_payload(
        native_cash_in_base(costs.commission_cost_cash, instrument, portfolio),
        native_cash_in_base(costs.slippage_cost_cash, instrument, portfolio),
        native_cash_in_base(costs.spread_cost_cash, instrument, portfolio),
        native_cash_in_base(costs.roll_cost_cash, instrument, portfolio),
        previous_equity_cash,
        roll_contract_turnover=roll_turnover,
        roll_turnover_notional=native_cash_in_base(
            roll_turnover_notional,
            instrument,
            portfolio,
        ),
    )


def spread_cost_cash_for_order(
    *,
    contract_delta: float,
    turnover: float,
    turnover_notional: float,
    instrument: IntradayEventInstrumentSpec,
    portfolio: IntradayEventPortfolioConfig,
    execution: IntradayEventExecutionConfig,
    previous_equity_cash: float,
) -> float:
    half_spread_bps = execution.bid_ask_spread_bps / 20_000.0
    half_spread_ticks = execution.bid_ask_spread_ticks / 2.0
    if portfolio.sizing_mode == "signal_units":
        spread_cost_cash = turnover * half_spread_bps * previous_equity_cash
    else:
        spread_cost_cash = turnover_notional * half_spread_bps
    tick_spread_cash = abs(contract_delta) * half_spread_ticks * effective_tick_value(instrument)
    if portfolio.sizing_mode == "signal_units":
        tick_spread_cash = native_cash_in_base(tick_spread_cash, instrument, portfolio)
    spread_cost_cash += tick_spread_cash
    return spread_cost_cash


def roll_cost_cash_for_event(
    row: dict[str, Any],
    next_row: dict[str, Any],
    current_contracts: float,
    exit_price: float,
    point_value: float,
    tick_value: float,
    execution: IntradayEventExecutionConfig,
) -> float:
    if str(row["active_contract"]) == str(next_row["active_contract"]):
        return 0.0
    if current_contracts == 0.0:
        return 0.0
    roll_notional = abs(current_contracts) * exit_price * point_value
    roll_cost_cash = roll_notional * execution.roll_cost_bps / 10_000.0
    return roll_cost_cash + abs(current_contracts) * execution.roll_slippage_ticks * tick_value


def cost_payload(
    commission_cost_cash: float,
    slippage_cost_cash: float,
    spread_cost_cash: float,
    roll_cost_cash: float,
    previous_equity_cash: float,
    *,
    roll_contract_turnover: float = 0.0,
    roll_turnover_notional: float = 0.0,
) -> dict[str, float]:
    total_cost_cash = commission_cost_cash + slippage_cost_cash + spread_cost_cash + roll_cost_cash
    return {
        "commission_cost_cash": commission_cost_cash,
        "slippage_cost_cash": slippage_cost_cash,
        "spread_cost_cash": spread_cost_cash,
        "roll_cost_cash": roll_cost_cash,
        "roll_contract_turnover": roll_contract_turnover,
        "roll_turnover_notional": roll_turnover_notional,
        "total_cost_cash": total_cost_cash,
        "total_cost_return": ratio_float(total_cost_cash, previous_equity_cash),
    }


def effective_tick_value(instrument: IntradayEventInstrumentSpec) -> float:
    return futures_effective_tick_value(futures_instrument_spec(instrument))


def futures_instrument_spec(instrument: IntradayEventInstrumentSpec) -> FuturesInstrumentSpec:
    return FuturesInstrumentSpec(
        symbol=instrument.symbol,
        contract_multiplier=instrument.contract_multiplier,
        lot_size=instrument.lot_size,
        tick_size=instrument.tick_size,
        tick_value=instrument.tick_value,
        initial_margin=instrument.initial_margin,
        exchange_fee_per_contract=instrument.exchange_fee_per_contract,
    )


def futures_portfolio_spec(
    portfolio: IntradayEventPortfolioConfig,
    instrument: IntradayEventInstrumentSpec | None = None,
) -> FuturesPortfolioSpec:
    def native(value: float | None) -> float | None:
        if value is None or instrument is None:
            return value
        return amount_from_base_currency(
            value,
            instrument.pnl_currency,
            portfolio.base_currency,
            portfolio.fx_rates,
        )

    return FuturesPortfolioSpec(
        initial_capital=portfolio.initial_capital,
        sizing_mode=portfolio.sizing_mode,
        fixed_contracts=portfolio.fixed_contracts,
        target_notional=native(portfolio.target_notional),
        allow_fractional_contracts=portfolio.allow_fractional_contracts,
        contract_rounding=portfolio.contract_rounding,
        max_position_notional=native(portfolio.max_position_notional),
        max_leverage=portfolio.max_leverage,
        max_margin_usage=portfolio.max_margin_usage,
        risk_policy=portfolio.risk_policy,
    )


def native_cash_in_base(
    amount: float,
    instrument: IntradayEventInstrumentSpec,
    portfolio: IntradayEventPortfolioConfig,
) -> float:
    return amount_in_base_currency(
        amount,
        instrument.pnl_currency,
        portfolio.base_currency,
        portfolio.fx_rates,
    )
