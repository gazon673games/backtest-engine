from __future__ import annotations

from typing import Any

from backtest_engine.accounting.futures import position_notional
from backtest_engine.accounting.ledger_runtime import amount_in_base_currency
from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig
from backtest_engine.portfolio.multi.futures import (
    futures_instrument_spec,
    margin_requirement,
)
from backtest_engine.portfolio.multi.interval_accounting import calculate_multi_interval


def rejected_rebalance_row(
    row: dict[str, Any],
    previous_target: float,
    previous_contracts: float,
    previous_equity_cash: float,
    config: MultiInstrumentBacktestConfig,
) -> dict[str, Any]:
    rejected = dict(row)
    fill_price = float(row["_reference_fill_price"])
    fill_instrument = row["_instrument"]
    accounting = calculate_multi_interval(
        fill_row=row["_fill_row"],
        exit_row=row["_exit_row"],
        target_contracts=previous_contracts,
        contract_delta=0.0,
        fill_price=fill_price,
        exit_price=float(row["exit_price"]),
        fill_instrument=fill_instrument,
        exit_instrument=row["_exit_instrument"],
        execution=config.execution,
    )
    required_margin = margin_requirement(previous_contracts, fill_instrument)
    required_margin_base = amount_in_base_currency(
        required_margin,
        fill_instrument.pnl_currency,
        config.portfolio.base_currency,
        config.portfolio.fx_rates,
    )
    net_pnl = accounting.gross_pnl - accounting.costs.total_cost_cash
    net_pnl_base = amount_in_base_currency(
        net_pnl,
        fill_instrument.pnl_currency,
        config.portfolio.base_currency,
        config.portfolio.fx_rates,
    )
    rejected.update(
        {
            "target": previous_target,
            "target_contracts": previous_contracts,
            "contract_delta": 0.0,
            "order_status": "rebalance_rejected",
            "fill_fraction": 0.0,
            "filled_contract_delta": 0.0,
            "fill_price": fill_price,
            "position_notional": position_notional(
                previous_contracts,
                fill_price,
                futures_instrument_spec(fill_instrument),
            ),
            "margin_requirement": required_margin,
            "margin_usage": (
                required_margin_base / previous_equity_cash if previous_equity_cash else None
            ),
            "position_turnover_notional": accounting.position_turnover_notional,
            "roll_turnover_notional": accounting.roll_turnover_notional,
            "roll_contract_turnover": accounting.roll_contract_turnover,
            "turnover_notional": accounting.turnover_notional,
            "roll_pnl_excluded": accounting.roll_pnl_excluded,
            "gross_pnl": accounting.gross_pnl,
            "cost_cash": accounting.costs.total_cost_cash,
            "commission_cost_cash": accounting.costs.commission_cost_cash,
            "slippage_cost_cash": accounting.costs.slippage_cost_cash,
            "spread_cost_cash": accounting.costs.spread_cost_cash,
            "roll_cost_cash": accounting.costs.roll_cost_cash,
            "net_pnl": net_pnl,
            "net_return": net_pnl_base / previous_equity_cash,
        }
    )
    return rejected
