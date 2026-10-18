from __future__ import annotations

from typing import Any

from backtest_engine.accounting.futures import (
    pnl_cash as futures_pnl_cash,
)
from backtest_engine.accounting.futures import (
    position_notional as futures_position_notional,
)
from backtest_engine.accounting.futures import (
    turnover_notional as futures_turnover_notional,
)
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig
from backtest_engine.intraday.event.futures import (
    contract_point_value,
    futures_instrument_spec,
    native_cash_in_base,
)
from backtest_engine.intraday.event.utils import ratio_float


def build_position_state(
    row: dict[str, Any],
    previous_signal: float,
    previous_contracts: float,
    current_signal: float,
    current_contracts: float,
) -> dict[str, float]:
    return {
        "close": float(row["close"]),
        "previous_signal": previous_signal,
        "previous_contracts": previous_contracts,
        "current_signal": current_signal,
        "current_contracts": current_contracts,
        "turnover": abs(current_signal - previous_signal),
        "contract_delta": current_contracts - previous_contracts,
    }


def build_instrument_state(
    instrument_spec: Any,
    fill_price: float,
    position_state: dict[str, float],
    config: IntradayEventBacktestConfig,
) -> dict[str, Any]:
    futures_instrument = futures_instrument_spec(instrument_spec)
    contract_delta = position_state["contract_delta"]
    current_contracts = position_state["current_contracts"]
    position_notional = futures_position_notional(
        current_contracts,
        fill_price,
        futures_instrument,
    )
    turnover_notional = futures_turnover_notional(
        contract_delta,
        fill_price,
        futures_instrument,
    )
    cash_to_base_rate = native_cash_in_base(1.0, instrument_spec, config.portfolio)
    return {
        "instrument_spec": instrument_spec,
        "futures_instrument": futures_instrument,
        "point_value": contract_point_value(instrument_spec) * cash_to_base_rate,
        "position_notional": position_notional * cash_to_base_rate,
        "turnover_notional": turnover_notional * cash_to_base_rate,
        "turnover_notional_native": turnover_notional,
        "cash_to_base_rate": cash_to_base_rate,
        "initial_margin": (
            None
            if instrument_spec.initial_margin is None
            else instrument_spec.initial_margin * cash_to_base_rate
        ),
    }


def build_pnl_state(
    *,
    config: IntradayEventBacktestConfig,
    current_signal: float,
    current_contracts: float,
    fill_price: float,
    exit_price: float,
    instrument_state: dict[str, Any],
    cost_parts: dict[str, float],
    previous_equity_cash: float,
    contract_roll: bool = False,
) -> dict[str, float]:
    validate_interval_prices(fill_price, exit_price)
    has_exposure = current_signal != 0.0 or current_contracts != 0.0
    if contract_roll and has_exposure and config.execution.roll_pnl_policy == "reject":
        raise DatasetValidationError(
            "Cannot calculate PnL across a futures roll from non-overlapping contract prices. "
            "Use roll_pnl_policy=exclude_gap only when the strategy is intentionally flat "
            "between the old-contract close and new-contract entry."
        )
    # Under the explicit exclude_gap policy the old contract is closed at the
    # interval start and the new one is opened at the interval end.
    pnl_exit_price = fill_price if contract_roll else exit_price
    asset_return = pnl_exit_price / fill_price - 1.0
    if config.portfolio.sizing_mode == "signal_units":
        return signal_units_pnl_state(
            current_signal,
            asset_return,
            cost_parts,
            previous_equity_cash,
        )
    return contracts_pnl_state(
        current_contracts,
        fill_price,
        pnl_exit_price,
        instrument_state,
        cost_parts,
        previous_equity_cash,
        asset_return,
    )


def signal_units_pnl_state(
    current_signal: float,
    asset_return: float,
    cost_parts: dict[str, float],
    previous_equity_cash: float,
) -> dict[str, float]:
    gross_return = current_signal * asset_return
    cost = cost_parts["total_cost_return"]
    net_return = gross_return - cost
    gross_pnl = gross_return * previous_equity_cash
    cost_cash = cost_parts["total_cost_cash"]
    net_pnl = net_return * previous_equity_cash
    return base_pnl_state(
        asset_return,
        gross_return,
        gross_pnl,
        cost,
        cost_cash,
        net_return,
        net_pnl,
    )


def contracts_pnl_state(
    current_contracts: float,
    fill_price: float,
    exit_price: float,
    instrument_state: dict[str, Any],
    cost_parts: dict[str, float],
    previous_equity_cash: float,
    asset_return: float,
) -> dict[str, float]:
    gross_pnl_native = futures_pnl_cash(
        contracts=current_contracts,
        fill_price=fill_price,
        exit_price=exit_price,
        instrument=instrument_state["futures_instrument"],
    )
    gross_pnl = gross_pnl_native * instrument_state["cash_to_base_rate"]
    cost_cash = cost_parts["total_cost_cash"]
    net_pnl = gross_pnl - cost_cash
    return base_pnl_state(
        asset_return,
        ratio_float(gross_pnl, previous_equity_cash),
        gross_pnl,
        ratio_float(cost_cash, previous_equity_cash),
        cost_cash,
        ratio_float(net_pnl, previous_equity_cash),
        net_pnl,
    )


def base_pnl_state(
    asset_return: float,
    gross_return: float,
    gross_pnl: float,
    cost: float,
    cost_cash: float,
    net_return: float,
    net_pnl: float,
) -> dict[str, float]:
    return {
        "asset_return": asset_return,
        "gross_return": gross_return,
        "gross_pnl": gross_pnl,
        "cost": cost,
        "cost_cash": cost_cash,
        "net_return": net_return,
        "net_pnl": net_pnl,
        "current_equity_cash": 0.0,
        "current_equity": 1.0,
    }


def validate_interval_prices(fill_price: float, exit_price: float) -> None:
    if fill_price == 0.0 or exit_price == 0.0:
        raise DatasetValidationError("close prices used for event returns must not be zero")
