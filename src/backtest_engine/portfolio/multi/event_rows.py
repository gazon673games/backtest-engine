from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from backtest_engine.accounting.futures import (
    position_notional as futures_position_notional,
)
from backtest_engine.accounting.ledger_runtime import amount_in_base_currency
from backtest_engine.execution.model import execution_interval
from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig
from backtest_engine.portfolio.multi.data import (
    instrument_spec_for_row,
    session_label_for_leg,
)
from backtest_engine.portfolio.multi.equity_rows import build_equity_row as build_equity_row
from backtest_engine.portfolio.multi.execution_v3 import LegExecutionBook, market_fill_result
from backtest_engine.portfolio.multi.futures import futures_instrument_spec
from backtest_engine.portfolio.multi.futures import (
    margin_requirement as instrument_margin_requirement,
)
from backtest_engine.portfolio.multi.interval_accounting import calculate_multi_interval
from backtest_engine.portfolio.multi.rejections import rejected_rebalance_row
from backtest_engine.portfolio.multi.utils import finite_float


def build_leg_rows_for_time(
    *,
    current_ts: datetime,
    tradable_assets: list[str],
    exact_indices: dict[str, dict[datetime, int]],
    rows_by_asset: dict[str, Sequence[dict[str, Any]]],
    loaded_legs: dict[str, dict[str, Any]],
    config: MultiInstrumentBacktestConfig,
    previous_targets_snapshot: dict[str, float],
    current_targets: dict[str, float],
    current_contracts: dict[str, float],
    target_targets: dict[str, float],
    target_contracts: dict[str, float],
    reasons: dict[str, str | None],
    features: dict[str, dict[str, Any]],
    risk_result: dict[str, Any],
    previous_equity_cash: float,
    execution_books: dict[str, LegExecutionBook] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for asset_id in tradable_assets:
        row = build_leg_row(
            asset_id=asset_id,
            current_ts=current_ts,
            exact_indices=exact_indices,
            rows_by_asset=rows_by_asset,
            loaded_legs=loaded_legs,
            config=config,
            previous_target=previous_targets_snapshot[asset_id],
            previous_contracts=current_contracts[asset_id],
            target=target_targets[asset_id],
            target_contracts=target_contracts[asset_id],
            reason=reasons.get(asset_id),
            features=features.get(asset_id, {}),
            risk_result=risk_result,
            previous_equity_cash=previous_equity_cash,
            book=execution_books.get(asset_id) if execution_books is not None else None,
            has_decision=asset_id in reasons,
        )
        rows.append(row)
    if config.execution.rebalance_policy == "all_or_none" and any(
        float(row["fill_fraction"]) < 1.0 for row in rows
    ):
        if execution_books is not None:
            for asset_id in tradable_assets:
                execution_books[asset_id].rollback_fill()
        rows = [
            rejected_rebalance_row(
                row,
                previous_targets_snapshot[str(row["asset_id"])],
                current_contracts[str(row["asset_id"])],
                previous_equity_cash,
                config,
            )
            for row in rows
        ]
    for row in rows:
        current_targets[str(row["asset_id"])] = float(row["target"])
        current_contracts[str(row["asset_id"])] = float(row["target_contracts"])
    return rows


def build_leg_row(
    *,
    asset_id: str,
    current_ts: datetime,
    exact_indices: dict[str, dict[datetime, int]],
    rows_by_asset: dict[str, Sequence[dict[str, Any]]],
    loaded_legs: dict[str, dict[str, Any]],
    config: MultiInstrumentBacktestConfig,
    previous_target: float,
    previous_contracts: float,
    target: float,
    target_contracts: float,
    reason: str | None,
    features: dict[str, Any],
    risk_result: dict[str, Any],
    previous_equity_cash: float,
    book: LegExecutionBook | None = None,
    has_decision: bool = True,
) -> dict[str, Any]:
    index = exact_indices[asset_id][current_ts]
    rows = rows_by_asset[asset_id]
    interval = execution_interval(rows, index, config.execution.timing, config.execution.data_mode)
    fill_row = rows[interval.fill_index]
    exit_row = rows[interval.exit_index]
    leg_config = loaded_legs[asset_id]["config"]
    close = finite_float(fill_row["close"], f"{asset_id}.close")
    instrument_spec = instrument_spec_for_row(fill_row, interval.fill_ts, leg_config)
    exit_instrument_spec = instrument_spec_for_row(exit_row, interval.exit_ts, leg_config)
    futures_instrument = futures_instrument_spec(instrument_spec)
    if book is not None:
        book.update_instrument(
            str(fill_row.get("active_contract", asset_id)),
            interval.fill_index,
            previous_contracts,
            previous_target,
        )
        if has_decision:
            book.request(
                target_contracts, target, previous_contracts, interval.fill_index, fill_row["ts"]
            )
        target_contracts, target = book.requested_contracts, book.requested_target
    fill_result = market_fill_result(
        rows=rows,
        index=index,
        base_fill_price=interval.fill_price,
        previous_contracts=previous_contracts,
        requested_contracts=target_contracts,
        config=config,
        execution_index=interval.fill_index,
        book=book,
    )
    target_contracts = fill_result.target_contracts
    contract_delta = fill_result.filled_contract_delta
    fill_price = fill_result.fill_price
    if book is not None:
        target = (
            book.requested_target * target_contracts / book.requested_contracts
            if book.requested_contracts
            else 0.0
        )
    position_notional = futures_position_notional(
        target_contracts,
        fill_price,
        futures_instrument,
    )
    required_margin = instrument_margin_requirement(target_contracts, instrument_spec)
    contract_roll = str(fill_row.get("active_contract")) != str(exit_row.get("active_contract"))
    accounting = calculate_multi_interval(
        fill_row=fill_row,
        exit_row=exit_row,
        target_contracts=target_contracts,
        contract_delta=contract_delta,
        fill_price=fill_price,
        exit_price=interval.exit_price,
        fill_instrument=instrument_spec,
        exit_instrument=exit_instrument_spec,
        execution=config.execution,
    )
    net_pnl = accounting.gross_pnl - accounting.costs.total_cost_cash
    net_pnl_base = amount_in_base_currency(
        net_pnl,
        instrument_spec.pnl_currency,
        config.portfolio.base_currency,
        config.portfolio.fx_rates,
    )
    required_margin_base = amount_in_base_currency(
        required_margin,
        instrument_spec.pnl_currency,
        config.portfolio.base_currency,
        config.portfolio.fx_rates,
    )
    return {
        "_instrument": instrument_spec,
        "_exit_instrument": exit_instrument_spec,
        "_fill_row": fill_row,
        "_exit_row": exit_row,
        "_reference_fill_price": interval.fill_price,
        "ts": current_ts,
        "next_ts": interval.exit_ts,
        "fill_ts": interval.fill_ts,
        "asset_id": asset_id,
        "close": close,
        "next_close": interval.exit_price,
        "fill_price": fill_price,
        "exit_price": interval.exit_price,
        "active_contract": str(fill_row.get("active_contract", asset_id)),
        "next_active_contract": str(exit_row.get("active_contract", asset_id)),
        "contract_roll": contract_roll,
        "roll_pnl_policy": config.execution.roll_pnl_policy,
        "roll_pnl_excluded": accounting.roll_pnl_excluded,
        "previous_target": previous_target,
        "target": target,
        "previous_contracts": previous_contracts,
        "target_contracts": target_contracts,
        "contract_delta": contract_delta,
        "order_status": fill_result.order_status,
        "fill_fraction": fill_result.fill_fraction,
        "filled_contract_delta": fill_result.filled_contract_delta,
        "position_notional": position_notional,
        "margin_requirement": required_margin,
        "margin_usage": (
            required_margin_base / previous_equity_cash if previous_equity_cash else None
        ),
        "position_turnover_notional": accounting.position_turnover_notional,
        "roll_turnover_notional": accounting.roll_turnover_notional,
        "roll_contract_turnover": accounting.roll_contract_turnover,
        "turnover_notional": accounting.turnover_notional,
        "gross_pnl": accounting.gross_pnl,
        "cost_cash": accounting.costs.total_cost_cash,
        "commission_cost_cash": accounting.costs.commission_cost_cash,
        "slippage_cost_cash": accounting.costs.slippage_cost_cash,
        "spread_cost_cash": accounting.costs.spread_cost_cash,
        "roll_cost_cash": accounting.costs.roll_cost_cash,
        "net_pnl": net_pnl,
        "net_return": net_pnl_base / previous_equity_cash,
        "strategy_reason": reason,
        "strategy_features": json.dumps(features, sort_keys=True, default=str),
        "risk_status": risk_result["status"],
        "risk_scale": risk_result["scale"],
        "session_label": session_label_for_leg(current_ts, leg_config),
        "execution_timing": config.execution.timing,
        "instrument_symbol": instrument_spec.symbol,
        "price_currency": instrument_spec.price_currency,
        "pnl_currency": instrument_spec.pnl_currency,
        "contract_multiplier": instrument_spec.contract_multiplier,
        "lot_size": instrument_spec.lot_size,
        "tick_size": instrument_spec.tick_size,
        "tick_value": instrument_spec.tick_value,
        "initial_margin": instrument_spec.initial_margin,
        "exchange_fee_per_contract": instrument_spec.exchange_fee_per_contract,
        "asset_class": instrument_spec.asset_class,
        "instrument_type": instrument_spec.instrument_type,
        "base_currency": instrument_spec.base_currency,
        "quote_currency": instrument_spec.quote_currency,
        "settlement_currency": instrument_spec.settlement_currency,
        "trading_fee_cash": 0.0,
        "funding_cost_cash": 0.0,
        "liquidation_fee_cash": 0.0,
        "maintenance_margin": 0.0,
        "liquidated": False,
        "liquidation_price": None,
        "bankruptcy_price": None,
        "liquidation_status": "none",
    }
