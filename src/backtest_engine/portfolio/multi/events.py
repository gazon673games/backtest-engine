"""Synchronizes multi-leg decisions, fills, risk, ledger updates, and snapshots."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

from backtest_engine.accounting.historical_rates import HistoricalRateBook
from backtest_engine.accounting.ledger_frames import ledger_frame_from_snapshots
from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime
from backtest_engine.core.risk_safety import assert_end_position
from backtest_engine.intraday.event.settlement import EventAccountingRuntime
from backtest_engine.portfolio.multi.accounting import validate_crypto_accounting_topology
from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig
from backtest_engine.portfolio.multi.cross_margin import SharedCrossMarginRuntime
from backtest_engine.portfolio.multi.event_step import process_event_time
from backtest_engine.portfolio.multi.execution_v3 import LegExecutionBook
from backtest_engine.portfolio.multi.lifecycle import finish_multi_strategy, start_multi_strategy
from backtest_engine.portfolio.multi.metrics import (
    metrics,
    portfolio_equity_frame,
    portfolio_legs_frame,
)
from backtest_engine.portfolio.multi.timeline import (
    event_times,
    exact_event_indices,
    timestamps_by_asset,
    validate_execution_timeline,
)
from backtest_engine.portfolio.multi.universe_runtime import MultiUniverseRuntime


def build_multi_instrument_events(
    config: MultiInstrumentBacktestConfig,
    loaded_legs: dict[str, dict[str, Any]],
    generate_targets: Callable[..., Any],
    order_history: list[dict[str, Any]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    shared_cross_margin = validate_crypto_accounting_topology(loaded_legs, config)
    frames = {asset_id: payload["frame"] for asset_id, payload in loaded_legs.items()}
    rows_by_asset = {asset_id: payload["rows"] for asset_id, payload in loaded_legs.items()}
    timestamps = timestamps_by_asset(frames)
    exact_indices = exact_event_indices(timestamps, config.execution.timing)
    validate_execution_timeline(exact_indices, rows_by_asset, config.execution.timing)
    current_targets = {asset_id: 0.0 for asset_id in frames}
    current_contracts = {asset_id: 0.0 for asset_id in frames}
    execution_books = (
        {asset_id: LegExecutionBook(config) for asset_id in frames}
        if config.execution.engine == "v3"
        else None
    )
    start_ts = min(ts for asset_ts in timestamps.values() for ts in asset_ts)
    rate_book = (
        HistoricalRateBook.from_path(config.portfolio.valuation_rates_path)
        if config.portfolio.valuation_rates_path is not None
        else None
    )
    static_rates = config.portfolio.fx_rates
    initial_rates = (
        rate_book.rates_at(start_ts, config.portfolio.fx_rates)
        if rate_book is not None
        else config.portfolio.fx_rates
    )
    ledger_runtime = BacktestLedgerRuntime(
        initial_capital=config.portfolio.initial_capital,
        base_currency=config.portfolio.base_currency,
        fx_rates=initial_rates,
        start_ts=start_ts,
        collateral_currency=config.portfolio.collateral_currency,
        initial_collateral=config.portfolio.initial_collateral,
        rate_book=rate_book,
        history_size=2 * len(frames) + 3,
    )
    ledger_runtime.static_rates = static_rates
    accounting_runtime = EventAccountingRuntime(ledger_runtime)
    shared_margin_runtime = (
        SharedCrossMarginRuntime(ledger_runtime) if shared_cross_margin else None
    )
    universe_runtime = MultiUniverseRuntime(config.universe, list(frames))
    start_multi_strategy(generate_targets, config.strategy.strategy_params)
    equity_cash = ledger_runtime.equity_cash
    leg_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for current_ts in event_times(timestamps, config):
        if rate_book is not None:
            rates = ledger_runtime.activate_rates(current_ts)
            config = config.model_copy(
                update={"portfolio": config.portfolio.model_copy(update={"fx_rates": rates})}
            )
        equity_cash = process_event_time(
            current_ts=current_ts,
            exact_indices=exact_indices,
            timestamps=timestamps,
            frames=frames,
            rows_by_asset=rows_by_asset,
            loaded_legs=loaded_legs,
            config=config,
            generate_targets=generate_targets,
            current_targets=current_targets,
            current_contracts=current_contracts,
            equity_cash=equity_cash,
            decision_equity_cash=ledger_runtime.decision_equity_at(current_ts),
            ledger_runtime=ledger_runtime,
            accounting_runtime=accounting_runtime,
            shared_margin_runtime=shared_margin_runtime,
            universe_runtime=universe_runtime,
            leg_rows=leg_rows,
            equity_rows=equity_rows,
            execution_books=execution_books,
        )

    cancelled_orders = sum(book.finish() for book in (execution_books or {}).values())
    if order_history is not None:
        for asset_id, book in (execution_books or {}).items():
            order_history.extend(book.history(asset_id))
    portfolio_legs = portfolio_legs_frame(leg_rows)
    portfolio_equity = portfolio_equity_frame(equity_rows)
    for asset_id, contracts in current_contracts.items():
        assert_end_position(
            contracts,
            config.execution.end_of_data_position_policy,
            label=asset_id,
        )
    finish_multi_strategy(
        generate_targets,
        event_count=portfolio_equity.height,
        equity_cash=equity_cash,
        positions=current_contracts,
    )
    return (
        portfolio_legs,
        portfolio_equity,
        ledger_frame_from_snapshots(ledger_runtime.snapshots),
        {
            **metrics(portfolio_equity, portfolio_legs, config),
            "pending_orders_cancelled": cancelled_orders,
        },
    )
