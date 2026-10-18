"""Reject optional position additions before creating an execution order."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from backtest_engine.accounting.fees import trading_fee_bps
from backtest_engine.accounting.futures import cost_breakdown
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.execution.bar_fills import impact_price, market_price
from backtest_engine.execution.models import OrderRequest, parse_execution_bar
from backtest_engine.intraday.event.config import (
    IntradayEventBacktestConfig,
    IntradayEventInstrumentSpec,
)
from backtest_engine.intraday.event.futures import (
    contract_point_value,
    futures_execution_cost_spec,
    futures_instrument_spec,
    native_cash_in_base,
)
from backtest_engine.strategies.plugins.normalization import IntradayEventPluginDecision


def skip_unaffordable_add(
    decision: IntradayEventPluginDecision,
    *,
    previous_contracts: float,
    equity_cash: float,
    fill_price: float,
    fill_row: dict[str, Any],
    instrument: IntradayEventInstrumentSpec,
    config: IntradayEventBacktestConfig,
    decision_state: dict[str, Any],
) -> bool:
    if not decision.skip_unaffordable_add or decision.contracts is None:
        return False
    requested = decision.target * decision.contracts
    if requested * previous_contracts <= 0 or abs(requested) <= abs(previous_contracts):
        return False
    execution = config.execution
    if (
        decision.order_type != "market"
        or decision.time_in_force != "ioc"
        or execution.latency_bars
        or execution.order_book_path is not None
        or instrument.instrument_type not in {"future", "perpetual"}
        or config.portfolio.sizing_mode == "signal_units"
    ):
        raise DatasetValidationError(
            "skip_unaffordable_add requires immediate IOC market orders in futures contract mode"
        )
    perpetual = instrument.instrument_type == "perpetual"
    if not perpetual and instrument.initial_margin is None:
        raise DatasetValidationError(
            f"skip_unaffordable_add requires initial_margin for {instrument.symbol}"
        )

    delta = requested - previous_contracts
    request = OrderRequest(
        order_id="affordability-preview",
        side="buy" if delta > 0 else "sell",
        quantity=abs(delta),
        order_type="market",
        time_in_force="ioc",
    )
    if execution.engine == "v3":
        bar = parse_execution_bar(
            {**fill_row, "market_reference_price": fill_price}, execution.data_mode
        )
        fill_price = impact_price(request, market_price(request, bar), execution.market_impact_bps)
    execution_costs = futures_execution_cost_spec(execution)
    if perpetual:
        execution_costs = replace(
            execution_costs,
            commission_bps=trading_fee_bps(
                instrument, execution.liquidity_role, execution.commission_bps
            ),
        )
    costs = cost_breakdown(
        contract_delta=abs(delta),
        turnover_notional_cash=abs(delta) * fill_price * contract_point_value(instrument),
        instrument=futures_instrument_spec(instrument),
        execution=execution_costs,
    )
    add_cost = native_cash_in_base(costs.total_cost_cash, instrument, config.portfolio)
    native_margin = (
        fill_price * contract_point_value(instrument) / config.portfolio.leverage
        if perpetual
        else instrument.initial_margin
    )
    margin = native_cash_in_base(native_margin, instrument, config.portfolio)
    required = abs(requested) * margin
    # Reserve the full addition and its costs, before any volume-based partial fill.
    budget = (equity_cash - add_cost) * min(config.portfolio.max_margin_usage or 1.0, 1.0)
    if required <= budget:
        return False

    reason = "insufficient_margin_add_skipped"
    decision_state.update(
        requested_signal=decision.target,
        requested_contracts=requested,
        strategy_reason=reason,
        signal_reason=reason,
        order_status="rejected_insufficient_margin",
        order_filled=False,
        order_type=decision.order_type,
        strategy_features={
            **decision.features,
            "requestedStrategyReason": decision.reason,
            "addAffordability": {
                "status": "skipped",
                "equityCash": equity_cash,
                "previousContracts": previous_contracts,
                "requestedContracts": requested,
                "requiredMargin": required,
                "estimatedAddCostCash": add_cost,
                "marginBudgetAfterCosts": budget,
            },
        },
    )
    return True
