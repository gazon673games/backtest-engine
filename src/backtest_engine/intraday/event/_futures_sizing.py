from __future__ import annotations

import math
from decimal import ROUND_FLOOR, Decimal
from typing import Any

from backtest_engine.accounting.futures import (
    apply_contract_risk,
)
from backtest_engine.accounting.futures import (
    contract_point_value as futures_contract_point_value,
)
from backtest_engine.accounting.futures import (
    target_contracts as futures_target_contracts,
)
from backtest_engine.accounting.ledger_runtime import (
    amount_from_base_currency,
)
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.intraday.event._futures_costs import (
    contract_costs,
    cost_payload,
    effective_tick_value,
    futures_instrument_spec,
    futures_portfolio_spec,
    native_cash_in_base,
    roll_cost_cash_for_event,
    spread_cost_cash_for_order,
)
from backtest_engine.intraday.event.config import (
    IntradayEventExecutionConfig,
    IntradayEventInstrumentSpec,
    IntradayEventPortfolioConfig,
)


def target_contracts(
    signal: float,
    fill_price: float,
    instrument: IntradayEventInstrumentSpec,
    portfolio: IntradayEventPortfolioConfig,
    equity_cash: float,
) -> float:
    if instrument.instrument_type == "spot" and portfolio.sizing_mode == "signal_units":
        if signal < 0.0:
            raise DatasetValidationError("Spot signal_units cannot request a short position")
        native_equity = amount_from_base_currency(
            equity_cash,
            instrument.quote_currency or instrument.pnl_currency,
            portfolio.base_currency,
            portfolio.fx_rates,
        )
        contracts = signal * native_equity / (fill_price * contract_point_value(instrument))
        return apply_risk_controls(
            contracts,
            fill_price,
            instrument,
            portfolio,
            equity_cash,
        )
    native_portfolio = futures_portfolio_spec(portfolio, instrument)
    contracts = futures_target_contracts(
        signal=signal,
        fill_price=fill_price,
        instrument=futures_instrument_spec(instrument),
        portfolio=native_portfolio,
        equity_cash=equity_cash,
    )
    return apply_risk_controls(contracts, fill_price, instrument, portfolio, equity_cash)


def apply_risk_controls(
    contracts: float,
    fill_price: float,
    instrument: IntradayEventInstrumentSpec,
    portfolio: IntradayEventPortfolioConfig,
    equity_cash: float,
) -> float:
    native_portfolio = futures_portfolio_spec(portfolio, instrument)
    native_equity = equity_cash
    if portfolio.max_leverage is not None or portfolio.max_margin_usage is not None:
        native_equity = amount_from_base_currency(
            equity_cash,
            instrument.pnl_currency,
            portfolio.base_currency,
            portfolio.fx_rates,
        )
    risk = apply_contract_risk(
        contracts=contracts,
        fill_price=fill_price,
        instrument=futures_instrument_spec(instrument),
        portfolio=native_portfolio,
        equity_cash=native_equity,
    )
    if risk.status == "rejected":
        max_contracts = 0.0 if risk.max_contracts is None else risk.max_contracts
        msg = f"target contracts {contracts} breach risk limit max_contracts={max_contracts:.12g}"
        raise DatasetValidationError(msg)
    if risk.status == "clipped" and instrument.quantity_step:
        # Risk caps are continuous; submitted quantities still obey the instrument's
        # historical step. Round down so quantization cannot increase exposure.
        step = Decimal(str(instrument.quantity_step))
        units = (Decimal(str(abs(risk.contracts))) / step).to_integral_value(ROUND_FLOOR)
        return math.copysign(float(units * step), risk.contracts)
    return risk.contracts


def round_contracts(value: float, rounding: str) -> float:
    if rounding == "floor":
        return float(math.floor(value))
    if rounding == "ceil":
        return float(math.ceil(value))
    return float(round(value))


def contract_point_value(instrument: IntradayEventInstrumentSpec) -> float:
    return futures_contract_point_value(futures_instrument_spec(instrument))


def execution_costs(
    *,
    row: dict[str, Any],
    next_row: dict[str, Any],
    current_contracts: float,
    contract_delta: float,
    turnover: float,
    turnover_notional: float,
    fill_price: float,
    exit_price: float,
    instrument: IntradayEventInstrumentSpec,
    portfolio: IntradayEventPortfolioConfig,
    execution: IntradayEventExecutionConfig,
    previous_equity_cash: float,
) -> dict[str, float]:
    point_value = contract_point_value(instrument)
    tick_value = effective_tick_value(instrument)
    if portfolio.sizing_mode != "signal_units":
        return contract_costs(
            row=row,
            next_row=next_row,
            current_contracts=current_contracts,
            contract_delta=contract_delta,
            turnover_notional=turnover_notional,
            fill_price=fill_price,
            exit_price=exit_price,
            instrument=instrument,
            portfolio=portfolio,
            execution=execution,
            previous_equity_cash=previous_equity_cash,
            point_value=point_value,
        )

    contract_roll = str(row["active_contract"]) != str(next_row["active_contract"])
    roll_turnover = 2.0 * abs(current_contracts) if contract_roll else 0.0
    total_turnover = turnover + roll_turnover
    cost_contract_delta = abs(contract_delta) + roll_turnover
    roll_turnover_notional = abs(current_contracts) * (fill_price + exit_price) * point_value
    if not contract_roll:
        roll_turnover_notional = 0.0
    commission_cost_cash = total_turnover * execution.commission_bps / 10_000.0
    commission_cost_cash *= previous_equity_cash
    slippage_cost_cash = total_turnover * execution.slippage_bps / 10_000.0 * previous_equity_cash
    tick_slippage_cash = cost_contract_delta * execution.slippage_ticks * tick_value
    slippage_cost_cash += native_cash_in_base(tick_slippage_cash, instrument, portfolio)
    spread_cost_cash = spread_cost_cash_for_order(
        contract_delta=cost_contract_delta,
        turnover=total_turnover,
        turnover_notional=turnover_notional + roll_turnover_notional,
        instrument=instrument,
        portfolio=portfolio,
        execution=execution,
        previous_equity_cash=previous_equity_cash,
    )
    roll_cost_cash = native_cash_in_base(
        roll_cost_cash_for_event(
            row,
            next_row,
            current_contracts,
            exit_price,
            point_value,
            tick_value,
            execution,
        ),
        instrument,
        portfolio,
    )
    return cost_payload(
        commission_cost_cash,
        slippage_cost_cash,
        spread_cost_cash,
        roll_cost_cash,
        previous_equity_cash,
        roll_contract_turnover=roll_turnover,
        roll_turnover_notional=native_cash_in_base(
            roll_turnover_notional,
            instrument,
            portfolio,
        ),
    )
