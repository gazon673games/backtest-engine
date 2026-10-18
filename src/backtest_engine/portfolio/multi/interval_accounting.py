from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backtest_engine.accounting.futures import (
    FuturesCostBreakdown,
    cost_breakdown,
    pnl_cash,
    position_notional,
    turnover_notional,
)
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.portfolio.multi.config import MultiInstrumentExecutionConfig
from backtest_engine.portfolio.multi.futures import (
    futures_execution_cost_spec,
    futures_instrument_spec,
)


@dataclass(frozen=True)
class MultiIntervalAccounting:
    gross_pnl: float
    costs: FuturesCostBreakdown
    position_turnover_notional: float
    roll_turnover_notional: float
    roll_contract_turnover: float
    roll_pnl_excluded: bool

    @property
    def turnover_notional(self) -> float:
        return self.position_turnover_notional + self.roll_turnover_notional


def calculate_multi_interval(
    *,
    fill_row: dict[str, Any],
    exit_row: dict[str, Any],
    target_contracts: float,
    contract_delta: float,
    fill_price: float,
    exit_price: float,
    fill_instrument: Any,
    exit_instrument: Any,
    execution: MultiInstrumentExecutionConfig,
) -> MultiIntervalAccounting:
    fill_spec = futures_instrument_spec(fill_instrument)
    exit_spec = futures_instrument_spec(exit_instrument)
    execution_spec = futures_execution_cost_spec(execution)
    contract_roll = str(fill_row.get("active_contract")) != str(exit_row.get("active_contract"))
    has_exposure = target_contracts != 0.0
    if contract_roll and has_exposure and execution.roll_pnl_policy == "reject":
        raise DatasetValidationError(
            "Cannot calculate multi-instrument PnL across a futures roll from "
            "non-overlapping contract prices; use roll_pnl_policy=exclude_gap "
            "to model an explicit close and reopen"
        )
    if contract_roll and has_exposure:
        _validate_roll_currency(fill_instrument, exit_instrument)

    position_turnover = turnover_notional(contract_delta, fill_price, fill_spec)
    base_costs = cost_breakdown(
        contract_delta=contract_delta,
        turnover_notional_cash=position_turnover,
        instrument=fill_spec,
        execution=execution_spec,
    )
    roll_turnover = 0.0
    roll_contract_turnover = 0.0
    roll_costs = _zero_costs()
    if contract_roll and has_exposure:
        old_notional = position_notional(target_contracts, fill_price, fill_spec)
        new_notional = position_notional(target_contracts, exit_price, exit_spec)
        roll_turnover = old_notional + new_notional
        roll_contract_turnover = 2.0 * abs(target_contracts)
        roll_costs = _roll_costs(
            target_contracts=target_contracts,
            old_notional=old_notional,
            new_notional=new_notional,
            fill_spec=fill_spec,
            exit_spec=exit_spec,
            execution_spec=execution_spec,
        )

    pnl_exit_price = fill_price if contract_roll else exit_price
    gross_pnl = pnl_cash(
        contracts=target_contracts,
        fill_price=fill_price,
        exit_price=pnl_exit_price,
        instrument=fill_spec,
    )
    return MultiIntervalAccounting(
        gross_pnl=gross_pnl,
        costs=_sum_costs(base_costs, roll_costs),
        position_turnover_notional=position_turnover,
        roll_turnover_notional=roll_turnover,
        roll_contract_turnover=roll_contract_turnover,
        roll_pnl_excluded=contract_roll and has_exposure,
    )


def _roll_costs(
    *,
    target_contracts: float,
    old_notional: float,
    new_notional: float,
    fill_spec: Any,
    exit_spec: Any,
    execution_spec: Any,
) -> FuturesCostBreakdown:
    close_costs = cost_breakdown(
        contract_delta=abs(target_contracts),
        turnover_notional_cash=old_notional,
        instrument=fill_spec,
        execution=execution_spec,
    )
    open_costs = cost_breakdown(
        contract_delta=abs(target_contracts),
        turnover_notional_cash=new_notional,
        instrument=exit_spec,
        execution=execution_spec,
    )
    explicit_roll_costs = cost_breakdown(
        contract_delta=0.0,
        turnover_notional_cash=0.0,
        current_contracts=target_contracts,
        roll_notional_cash=new_notional,
        instrument=exit_spec,
        execution=execution_spec,
    )
    return _sum_costs(close_costs, open_costs, explicit_roll_costs)


def _sum_costs(*parts: FuturesCostBreakdown) -> FuturesCostBreakdown:
    return FuturesCostBreakdown(
        commission_cost_cash=sum(part.commission_cost_cash for part in parts),
        slippage_cost_cash=sum(part.slippage_cost_cash for part in parts),
        spread_cost_cash=sum(part.spread_cost_cash for part in parts),
        roll_cost_cash=sum(part.roll_cost_cash for part in parts),
    )


def _zero_costs() -> FuturesCostBreakdown:
    return FuturesCostBreakdown(0.0, 0.0, 0.0, 0.0)


def _validate_roll_currency(fill_instrument: Any, exit_instrument: Any) -> None:
    if fill_instrument.pnl_currency != exit_instrument.pnl_currency:
        raise DatasetValidationError(
            "A futures roll cannot change pnl_currency within one portfolio leg"
        )
