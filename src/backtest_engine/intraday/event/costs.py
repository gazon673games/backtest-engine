"""Dispatches event costs to the asset-specific accounting model."""

from __future__ import annotations

from typing import Any

from backtest_engine.accounting.crypto import CryptoSettlement
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig
from backtest_engine.intraday.event.crypto import apply_crypto_costs
from backtest_engine.intraday.event.futures import contract_point_value, execution_costs


def calculate_event_costs(
    *,
    fill_row: dict[str, Any],
    exit_row: dict[str, Any],
    fill_price: float,
    exit_price: float,
    current_contracts: float,
    position_state: dict[str, float],
    instrument_state: dict[str, Any],
    instrument: Any,
    config: IntradayEventBacktestConfig,
    previous_equity_cash: float,
) -> tuple[dict[str, float], CryptoSettlement | None]:
    parts = execution_costs(
        row=fill_row,
        next_row=exit_row,
        current_contracts=current_contracts,
        contract_delta=position_state["contract_delta"],
        turnover=position_state["turnover"],
        turnover_notional=instrument_state["turnover_notional_native"],
        fill_price=fill_price,
        exit_price=exit_price,
        instrument=instrument,
        portfolio=config.portfolio,
        execution=config.execution,
        previous_equity_cash=previous_equity_cash,
    )
    return apply_crypto_costs(
        exit_row=exit_row,
        fill_price=fill_price,
        exit_price=exit_price,
        current_contracts=current_contracts,
        contract_delta=position_state["contract_delta"],
        point_value_native=contract_point_value(instrument),
        instrument=instrument,
        execution=config.execution,
        portfolio=config.portfolio,
        previous_equity_cash=previous_equity_cash,
        cost_parts=parts,
    )
