from __future__ import annotations

from typing import Any

from backtest_engine.accounting.futures import (
    FuturesExecutionCostSpec,
    FuturesInstrumentSpec,
    FuturesPortfolioSpec,
)
from backtest_engine.portfolio.multi.config import (
    MultiInstrumentExecutionConfig,
    MultiInstrumentInstrumentSpec,
    MultiInstrumentLegConfig,
    MultiInstrumentPortfolioConfig,
)


def futures_instrument_spec(
    instrument: MultiInstrumentInstrumentSpec | MultiInstrumentLegConfig,
) -> FuturesInstrumentSpec:
    return FuturesInstrumentSpec(
        symbol=getattr(instrument, "symbol", getattr(instrument, "asset_id", "*")),
        contract_multiplier=instrument.contract_multiplier,
        lot_size=instrument.lot_size,
        tick_size=instrument.tick_size,
        tick_value=instrument.tick_value,
        initial_margin=instrument.initial_margin,
        exchange_fee_per_contract=instrument.exchange_fee_per_contract,
    )


def futures_portfolio_spec(portfolio: MultiInstrumentPortfolioConfig) -> FuturesPortfolioSpec:
    return FuturesPortfolioSpec(
        initial_capital=portfolio.initial_capital,
        sizing_mode=portfolio.sizing_mode,
        fixed_contracts=portfolio.fixed_contracts,
        target_notional=portfolio.target_notional,
        allow_fractional_contracts=portfolio.allow_fractional_contracts,
        contract_rounding=portfolio.contract_rounding,
        max_position_notional=portfolio.max_position_notional,
        max_leverage=portfolio.max_leverage,
        max_margin_usage=portfolio.max_margin_usage,
        risk_policy=portfolio.risk_policy,
    )


def futures_execution_cost_spec(
    execution: MultiInstrumentExecutionConfig,
) -> FuturesExecutionCostSpec:
    return FuturesExecutionCostSpec(
        commission_bps=execution.commission_bps,
        slippage_bps=execution.slippage_bps,
        slippage_ticks=execution.slippage_ticks,
        bid_ask_spread_bps=execution.bid_ask_spread_bps,
        bid_ask_spread_ticks=execution.bid_ask_spread_ticks,
        roll_cost_bps=execution.roll_cost_bps,
        roll_slippage_ticks=execution.roll_slippage_ticks,
    )


def margin_requirement(contracts: float, instrument: Any) -> float:
    if contracts == 0.0 or instrument.initial_margin is None:
        return 0.0
    return abs(contracts) * instrument.initial_margin
