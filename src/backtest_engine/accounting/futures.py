"""Calculates contract sizing, notional, margin, PnL, fees, and risk limits."""

from __future__ import annotations

import math
from dataclasses import dataclass

from backtest_engine.core.exceptions import DatasetValidationError

FUTURES_SIZING_MODES = frozenset(
    {"signal_units", "fixed_contracts", "notional", "strategy_contracts"}
)
FUTURES_CONTRACT_ROUNDING = frozenset({"floor", "round", "ceil"})
FUTURES_RISK_POLICIES = frozenset({"reject", "clip"})


@dataclass(frozen=True)
class FuturesInstrumentSpec:
    symbol: str = "*"
    contract_multiplier: float = 1.0
    lot_size: float = 1.0
    tick_size: float | None = None
    tick_value: float | None = None
    initial_margin: float | None = None
    max_contracts: float | None = None
    exchange_fee_per_contract: float = 0.0


@dataclass(frozen=True)
class FuturesPortfolioSpec:
    initial_capital: float = 1.0
    sizing_mode: str = "fixed_contracts"
    fixed_contracts: float = 1.0
    target_notional: float | None = None
    allow_fractional_contracts: bool = True
    contract_rounding: str = "floor"
    max_position_notional: float | None = None
    max_leverage: float | None = None
    max_margin_usage: float | None = None
    risk_policy: str = "reject"


@dataclass(frozen=True)
class FuturesExecutionCostSpec:
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    slippage_ticks: float = 0.0
    bid_ask_spread_bps: float = 0.0
    bid_ask_spread_ticks: float = 0.0
    roll_cost_bps: float = 0.0
    roll_slippage_ticks: float = 0.0


@dataclass(frozen=True)
class FuturesCostBreakdown:
    commission_cost_cash: float
    slippage_cost_cash: float
    spread_cost_cash: float
    roll_cost_cash: float

    @property
    def total_cost_cash(self) -> float:
        return (
            self.commission_cost_cash
            + self.slippage_cost_cash
            + self.spread_cost_cash
            + self.roll_cost_cash
        )


@dataclass(frozen=True)
class FuturesRiskResult:
    contracts: float
    status: str
    scale: float
    max_contracts: float | None = None


def contract_point_value(instrument: FuturesInstrumentSpec) -> float:
    _require_positive(instrument.contract_multiplier, "contract_multiplier")
    _require_positive(instrument.lot_size, "lot_size")
    if instrument.tick_size is not None and instrument.tick_value is not None:
        tick_size = _require_positive(instrument.tick_size, "tick_size")
        tick_value = _require_positive(instrument.tick_value, "tick_value")
        return tick_value / tick_size
    return instrument.contract_multiplier * instrument.lot_size


def effective_tick_value(instrument: FuturesInstrumentSpec) -> float:
    if instrument.tick_value is not None:
        _require_positive(instrument.tick_value, "tick_value")
        return instrument.tick_value
    if instrument.tick_size is None:
        return 0.0
    _require_positive(instrument.tick_size, "tick_size")
    return instrument.tick_size * contract_point_value(instrument)


def position_notional(contracts: float, price: float, instrument: FuturesInstrumentSpec) -> float:
    return abs(contracts) * _require_positive(price, "price") * contract_point_value(instrument)


def turnover_notional(
    contract_delta: float,
    price: float,
    instrument: FuturesInstrumentSpec,
) -> float:
    return (
        abs(contract_delta) * _require_positive(price, "price") * contract_point_value(instrument)
    )


def target_contracts(
    *,
    signal: float,
    fill_price: float,
    instrument: FuturesInstrumentSpec,
    portfolio: FuturesPortfolioSpec,
    equity_cash: float,
) -> float:
    _validate_portfolio(portfolio)
    if signal == 0.0:
        return 0.0
    direction = 1.0 if signal > 0.0 else -1.0
    abs_signal = abs(signal)
    if portfolio.sizing_mode == "signal_units":
        contracts = signal
    elif portfolio.sizing_mode == "fixed_contracts":
        contracts = direction * portfolio.fixed_contracts * abs_signal
    elif portfolio.sizing_mode == "notional":
        if portfolio.target_notional is None:
            raise DatasetValidationError("target_notional is required for notional sizing")
        raw_contracts = (
            portfolio.target_notional
            * abs_signal
            / (_require_positive(fill_price, "fill_price") * contract_point_value(instrument))
        )
        contracts = direction * _round_contracts(raw_contracts, portfolio.contract_rounding)
    else:
        raise DatasetValidationError(f"Unsupported sizing mode: {portfolio.sizing_mode}")
    if not portfolio.allow_fractional_contracts:
        contracts = math.copysign(
            _round_contracts(abs(contracts), portfolio.contract_rounding),
            contracts,
        )
    return contracts


def apply_contract_risk(
    *,
    contracts: float,
    fill_price: float,
    instrument: FuturesInstrumentSpec,
    portfolio: FuturesPortfolioSpec,
    equity_cash: float,
) -> FuturesRiskResult:
    _validate_portfolio(portfolio)
    if equity_cash <= 0.0 and contracts != 0.0:
        return FuturesRiskResult(
            contracts=0.0 if portfolio.risk_policy == "clip" else contracts,
            status="clipped" if portfolio.risk_policy == "clip" else "rejected",
            scale=0.0,
            max_contracts=0.0,
        )
    limits = [] if instrument.max_contracts is None else [instrument.max_contracts]
    point_value = contract_point_value(instrument)
    if portfolio.max_position_notional is not None:
        limits.append(portfolio.max_position_notional / (fill_price * point_value))
    if portfolio.max_leverage is not None:
        limits.append((equity_cash * portfolio.max_leverage) / (fill_price * point_value))
    if portfolio.max_margin_usage is not None:
        if instrument.initial_margin is None:
            raise DatasetValidationError("initial_margin is required for max_margin_usage")
        limits.append((equity_cash * portfolio.max_margin_usage) / instrument.initial_margin)
    if not limits:
        return FuturesRiskResult(contracts=contracts, status="ok", scale=1.0)
    max_contracts = max(min(limits), 0.0)
    if abs(contracts) <= max_contracts:
        return FuturesRiskResult(
            contracts=contracts,
            status="ok",
            scale=1.0,
            max_contracts=max_contracts,
        )
    if portfolio.risk_policy == "clip":
        clipped = math.copysign(max_contracts, contracts)
        if not portfolio.allow_fractional_contracts:
            clipped = math.copysign(
                _round_contracts(abs(clipped), portfolio.contract_rounding),
                clipped,
            )
        return FuturesRiskResult(
            contracts=clipped,
            status="clipped",
            scale=0.0 if contracts == 0.0 else abs(clipped / contracts),
            max_contracts=max_contracts,
        )
    return FuturesRiskResult(
        contracts=contracts,
        status="rejected",
        scale=0.0,
        max_contracts=max_contracts,
    )


def pnl_cash(
    *,
    contracts: float,
    fill_price: float,
    exit_price: float,
    instrument: FuturesInstrumentSpec,
) -> float:
    return contracts * (exit_price - fill_price) * contract_point_value(instrument)


def cost_breakdown(
    *,
    contract_delta: float,
    turnover_notional_cash: float,
    current_contracts: float = 0.0,
    roll_notional_cash: float = 0.0,
    instrument: FuturesInstrumentSpec,
    execution: FuturesExecutionCostSpec,
) -> FuturesCostBreakdown:
    tick_value = effective_tick_value(instrument)
    commission_cost_cash = turnover_notional_cash * execution.commission_bps / 10_000.0
    commission_cost_cash += abs(contract_delta) * instrument.exchange_fee_per_contract
    slippage_cost_cash = turnover_notional_cash * execution.slippage_bps / 10_000.0
    slippage_cost_cash += abs(contract_delta) * execution.slippage_ticks * tick_value
    spread_cost_cash = turnover_notional_cash * execution.bid_ask_spread_bps / 20_000.0
    spread_cost_cash += abs(contract_delta) * (execution.bid_ask_spread_ticks / 2.0) * tick_value
    roll_cost_cash = roll_notional_cash * execution.roll_cost_bps / 10_000.0
    # A held position is not a roll. The caller supplies notional only on a roll.
    if roll_notional_cash > 0.0:
        roll_cost_cash += abs(current_contracts) * execution.roll_slippage_ticks * tick_value
    return FuturesCostBreakdown(
        commission_cost_cash=commission_cost_cash,
        slippage_cost_cash=slippage_cost_cash,
        spread_cost_cash=spread_cost_cash,
        roll_cost_cash=roll_cost_cash,
    )


def leverage(gross_notional: float, equity_cash: float) -> float | None:
    if equity_cash <= 0.0:
        return None
    return gross_notional / equity_cash


def _round_contracts(value: float, rounding: str) -> float:
    if rounding == "floor":
        return float(math.floor(value))
    if rounding == "ceil":
        return float(math.ceil(value))
    if rounding == "round":
        return float(round(value))
    raise DatasetValidationError(
        f"Invalid contract rounding: {rounding}. "
        f"Expected one of: {', '.join(sorted(FUTURES_CONTRACT_ROUNDING))}"
    )


def _validate_portfolio(portfolio: FuturesPortfolioSpec) -> None:
    if portfolio.sizing_mode not in FUTURES_SIZING_MODES:
        raise DatasetValidationError(
            f"Invalid sizing mode: {portfolio.sizing_mode}. "
            f"Expected one of: {', '.join(sorted(FUTURES_SIZING_MODES))}"
        )
    if portfolio.contract_rounding not in FUTURES_CONTRACT_ROUNDING:
        raise DatasetValidationError(
            f"Invalid contract rounding: {portfolio.contract_rounding}. "
            f"Expected one of: {', '.join(sorted(FUTURES_CONTRACT_ROUNDING))}"
        )
    if portfolio.risk_policy not in FUTURES_RISK_POLICIES:
        raise DatasetValidationError(
            f"Invalid risk policy: {portfolio.risk_policy}. "
            f"Expected one of: {', '.join(sorted(FUTURES_RISK_POLICIES))}"
        )


def _require_positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise DatasetValidationError(f"{name} must be positive and finite")
    return value
