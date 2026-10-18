from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import polars as pl

from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime
from backtest_engine.data.bar_availability import row_open_ts
from backtest_engine.intraday.event.settlement import EventAccountingRuntime
from backtest_engine.portfolio.multi.accounting import settle_mixed_leg_rows
from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig
from backtest_engine.portfolio.multi.context import MultiInstrumentStrategyContext
from backtest_engine.portfolio.multi.cross_margin import SharedCrossMarginRuntime
from backtest_engine.portfolio.multi.data import row_session_eligible_for_leg
from backtest_engine.portfolio.multi.event_rows import build_equity_row, build_leg_rows_for_time
from backtest_engine.portfolio.multi.execution_v3 import LegExecutionBook
from backtest_engine.portfolio.multi.lifecycle import notify_multi_fills
from backtest_engine.portfolio.multi.planning import (
    requested_contract_plan,
    scaled_targets,
    visible_indices_at,
)
from backtest_engine.portfolio.multi.plugin import generate_portfolio_decisions
from backtest_engine.portfolio.multi.risk import apply_portfolio_risk
from backtest_engine.portfolio.multi.risk_safety import (
    validate_after_settlement,
    validate_post_fill_rows,
)
from backtest_engine.portfolio.multi.universe_runtime import MultiUniverseRuntime
from backtest_engine.portfolio.multi.utils import as_datetime


def process_event_time(
    *,
    current_ts: datetime,
    exact_indices: dict[str, dict[datetime, int]],
    timestamps: dict[str, list[datetime]],
    frames: dict[str, pl.DataFrame],
    rows_by_asset: dict[str, Sequence[dict[str, Any]]],
    loaded_legs: dict[str, dict[str, Any]],
    config: MultiInstrumentBacktestConfig,
    generate_targets: Callable[..., Any],
    current_targets: dict[str, float],
    current_contracts: dict[str, float],
    equity_cash: float,
    decision_equity_cash: float,
    ledger_runtime: BacktestLedgerRuntime,
    accounting_runtime: EventAccountingRuntime,
    shared_margin_runtime: SharedCrossMarginRuntime | None,
    universe_runtime: MultiUniverseRuntime,
    leg_rows: list[dict[str, Any]],
    equity_rows: list[dict[str, Any]],
    execution_books: dict[str, LegExecutionBook] | None = None,
) -> float:
    exact_assets = [
        asset_id for asset_id, indices in exact_indices.items() if current_ts in indices
    ]
    decision_assets = decision_assets_for_time(
        current_ts,
        exact_assets,
        exact_indices,
        rows_by_asset,
        loaded_legs,
    )
    route = universe_runtime.route(
        current_ts,
        exact_assets,
        decision_assets,
        current_contracts,
    )
    tradable_assets = list(route.execution_assets)
    for asset_id in exact_assets:
        book = (execution_books or {}).get(asset_id)
        if book is None or not book.simulator.open_order_ids:
            continue
        if asset_id not in route.universe_assets:
            book.cancel_pending(exact_indices[asset_id][current_ts], "universe_removed")
            book.requested_contracts = current_contracts[asset_id]
            book.requested_target = current_targets[asset_id]
        elif asset_id not in tradable_assets:
            tradable_assets.append(asset_id)
    if not tradable_assets:
        return equity_cash
    visible_indices = visible_indices_at(timestamps, current_ts)
    previous_targets_snapshot = dict(current_targets)
    decisions = portfolio_decisions(
        current_ts,
        frames,
        visible_indices,
        route,
        current_targets,
        current_contracts,
        decision_equity_cash,
        generate_targets,
        config,
    )
    requested_targets, requested_contracts, reasons, features = requested_contract_plan(
        current_ts=current_ts,
        tradable_assets=tradable_assets,
        decisions=decisions,
        exact_indices=exact_indices,
        rows_by_asset=rows_by_asset,
        loaded_legs=loaded_legs,
        config=config,
        current_targets=current_targets,
        current_contracts=current_contracts,
        equity_cash=decision_equity_cash,
    )
    risk_result = apply_portfolio_risk(
        requested_contracts=requested_contracts,
        previous_contracts=current_contracts,
        tradable_assets=tradable_assets,
        frames=frames,
        visible_indices=visible_indices,
        loaded_legs=loaded_legs,
        equity_cash=decision_equity_cash,
        portfolio=config.portfolio,
    )
    target_contracts = risk_result["target_contracts"]
    target_targets = scaled_targets(
        tradable_assets,
        requested_targets,
        requested_contracts,
        target_contracts,
        current_targets,
        current_contracts,
    )
    timestamp_leg_rows = build_leg_rows_for_time(
        current_ts=current_ts,
        tradable_assets=tradable_assets,
        exact_indices=exact_indices,
        rows_by_asset=rows_by_asset,
        loaded_legs=loaded_legs,
        config=config,
        previous_targets_snapshot=previous_targets_snapshot,
        current_targets=current_targets,
        current_contracts=current_contracts,
        target_targets=target_targets,
        target_contracts=target_contracts,
        reasons=reasons,
        features=features,
        risk_result=risk_result,
        previous_equity_cash=equity_cash,
        execution_books=execution_books,
    )
    margin = validate_post_fill_rows(timestamp_leg_rows, config, ledger_runtime, equity_cash)
    previous_equity_cash = equity_cash
    equity_cash = settle_mixed_leg_rows(
        rows=timestamp_leg_rows,
        config=config,
        ledger=ledger_runtime,
        accounting=accounting_runtime,
        shared_margin=shared_margin_runtime,
        previous_equity_cash=previous_equity_cash,
        current_contracts=current_contracts,
        current_targets=current_targets,
    )
    for row in timestamp_leg_rows:
        row["net_return"] = float(row["net_pnl_base"]) / previous_equity_cash
    validate_after_settlement(equity_cash, margin)
    notify_multi_fills(generate_targets, timestamp_leg_rows)
    settlement_notifier = getattr(generate_targets, "notify_settlement", None)
    if settlement_notifier is not None:
        settlement_notifier(timestamp_leg_rows, equity_cash)
    leg_rows.extend(timestamp_leg_rows)
    equity_rows.append(
        build_equity_row(
            current_ts=max(as_datetime(row["next_ts"]) for row in timestamp_leg_rows),
            timestamp_leg_rows=timestamp_leg_rows,
            equity_cash=equity_cash,
            previous_equity_cash=previous_equity_cash,
            current_contracts=current_contracts,
            frames=frames,
            visible_indices=visible_indices,
            loaded_legs=loaded_legs,
            config=config,
            risk_result=risk_result,
        )
    )
    return equity_cash


def decision_assets_for_time(
    current_ts: datetime,
    exact_assets: list[str],
    exact_indices: dict[str, dict[datetime, int]],
    rows_by_asset: dict[str, Sequence[dict[str, Any]]],
    loaded_legs: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        asset_id
        for asset_id in exact_assets
        if row_session_eligible_for_leg(
            row_open_ts(rows_by_asset[asset_id][exact_indices[asset_id][current_ts]]),
            loaded_legs[asset_id]["config"],
        )
    ]


def portfolio_decisions(
    current_ts: datetime,
    frames: dict[str, pl.DataFrame],
    visible_indices: dict[str, int],
    route: Any,
    current_targets: dict[str, float],
    current_contracts: dict[str, float],
    equity_cash: float,
    generate_targets: Callable[..., Any],
    config: MultiInstrumentBacktestConfig,
) -> dict[str, dict[str, Any]]:
    generated = generate_portfolio_decisions(
        generate_targets,
        MultiInstrumentStrategyContext(
            current_ts=current_ts,
            frames=frames,
            visible_indices=visible_indices,
            tradable_assets=route.strategy_assets,
            previous_targets=current_targets,
            previous_contracts=current_contracts,
            equity_cash=equity_cash,
            universe_assets=route.universe_assets,
            universe_added=route.added,
            universe_removed=route.removed,
        ),
        config.strategy.strategy_params,
    )
    decisions = {
        asset_id: decision
        for asset_id, decision in generated.items()
        if asset_id in route.strategy_assets
    }
    decisions.update(route.forced_decisions)
    return decisions
