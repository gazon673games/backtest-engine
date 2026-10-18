"""Builds portfolio-level equity snapshots from executed leg rows."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import polars as pl

from backtest_engine.accounting.futures import leverage
from backtest_engine.accounting.ledger_runtime import amount_in_base_currency
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig
from backtest_engine.portfolio.multi.risk import gross_notional, portfolio_margin_requirement


def build_equity_row(
    *,
    current_ts: datetime,
    timestamp_leg_rows: list[dict[str, Any]],
    equity_cash: float,
    previous_equity_cash: float,
    current_contracts: dict[str, float],
    frames: dict[str, pl.DataFrame],
    visible_indices: dict[str, int],
    loaded_legs: dict[str, dict[str, Any]],
    config: MultiInstrumentBacktestConfig,
    risk_result: dict[str, Any],
) -> dict[str, Any]:
    timestamp_gross_pnl = sum(base_cash(row, "gross_pnl", config) for row in timestamp_leg_rows)
    timestamp_cost_cash = sum(base_cash(row, "cost_cash", config) for row in timestamp_leg_rows)
    converted_net_pnl = timestamp_gross_pnl - timestamp_cost_cash
    ledger_net_pnl = equity_cash - previous_equity_cash
    if not math.isclose(converted_net_pnl, ledger_net_pnl, rel_tol=1e-12, abs_tol=1e-9):
        raise DatasetValidationError(
            "Portfolio cash-flow reconciliation failed: "
            f"converted_net_pnl={converted_net_pnl}, ledger_net_pnl={ledger_net_pnl}"
        )
    portfolio_notional = gross_notional(
        contracts=current_contracts,
        frames=frames,
        visible_indices=visible_indices,
        loaded_legs=loaded_legs,
        portfolio=config.portfolio,
    )
    required_margin = portfolio_margin_requirement(
        contracts=current_contracts,
        frames=frames,
        visible_indices=visible_indices,
        loaded_legs=loaded_legs,
        portfolio=config.portfolio,
    )
    return {
        "ts": current_ts,
        "gross_pnl": timestamp_gross_pnl,
        "cost_cash": timestamp_cost_cash,
        "net_pnl": ledger_net_pnl,
        "equity_cash": equity_cash,
        "equity": equity_cash / config.portfolio.initial_capital,
        "net_return": ledger_net_pnl / previous_equity_cash,
        "gross_position_notional": portfolio_notional,
        "margin_requirement": required_margin,
        "margin_usage": required_margin / equity_cash if equity_cash else None,
        "leverage": leverage(portfolio_notional, equity_cash),
        "active_leg_count": sum(1 for value in current_contracts.values() if value != 0.0),
        "risk_status": risk_result["status"],
        "risk_scale": risk_result["scale"],
    }


def base_cash(
    row: dict[str, Any],
    column: str,
    config: MultiInstrumentBacktestConfig,
) -> float:
    if column + "_base" in row:
        return float(row[column + "_base"])
    currency = str(row.get("pnl_currency") or config.portfolio.base_currency)
    return amount_in_base_currency(
        float(row[column]),
        currency,
        config.portfolio.base_currency,
        config.portfolio.fx_rates,
    )
