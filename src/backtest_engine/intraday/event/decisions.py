from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.execution.model import execution_interval
from backtest_engine.intraday.event.add_affordability import skip_unaffordable_add
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig
from backtest_engine.intraday.event.execution_v3 import IntradayExecutionBookV3
from backtest_engine.intraday.event.futures import (
    apply_risk_controls,
    contract_point_value,
    native_cash_in_base,
    target_contracts,
)
from backtest_engine.intraday.event.orders import resolve_order_fill
from backtest_engine.intraday.event.state import CallSchedule
from backtest_engine.strategies.intraday_api import IndicatorCache, IntradayStrategyContext
from backtest_engine.strategies.plugins.loader import (
    generate_intraday_event_plugin_decision_from_callable,
)


def initial_decision_state(
    index: int,
    call_schedule: CallSchedule,
    current_signal: float,
    current_contracts: float,
    config: IntradayEventBacktestConfig,
    *,
    strategy_eligible: bool = True,
) -> dict[str, Any]:
    missing_decision = index in call_schedule.missing_closes
    return {
        "missing_decision": missing_decision,
        "missing_decision_policy": config.execution.missing_decision_policy
        if missing_decision
        else None,
        "strategy_called": (
            False if missing_decision else strategy_eligible and call_schedule.should_call(index)
        ),
        "strategy_reason": None,
        "strategy_features": {},
        "requested_signal": 0.0 if missing_decision else current_signal,
        "requested_contracts": 0.0 if missing_decision else current_contracts,
        "order_type": "market",
        "order_status": "not_required",
        "order_filled": False,
        "order_limit_price": None,
        "order_stop_price": None,
        "order_time_in_force": "ioc",
        "order_expires_after_bars": None,
        "order_trailing_offset": None,
        "order_oco_group": None,
        "order_id": None,
        "cancel_order_id": None,
        "replace_order_id": None,
        "take_profit_price": None,
        "stop_loss_price": None,
        "requested_contract_delta": 0.0,
        "filled_contract_delta": 0.0,
        "fill_fraction": 1.0,
        "volume_limit_contracts": None,
        "volume_limited": False,
        "signal_reason": "missing_decision_close" if missing_decision else None,
    }


def apply_strategy_decision(
    *,
    generate_signal: Any,
    df: pl.DataFrame,
    rows: Sequence[dict[str, Any]],
    index: int,
    dataset_validation: dict[str, Any],
    config: IntradayEventBacktestConfig,
    call_time: Any,
    previous_signal: float,
    previous_contracts: float,
    fill_price: float,
    row: dict[str, Any],
    next_row: dict[str, Any],
    instrument_spec: Any,
    equity_cash: float,
    decision_equity_cash: float,
    decision_state: dict[str, Any],
    decision_cash: dict[str, float] | None = None,
    execution_book: IntradayExecutionBookV3 | None = None,
    indicator_cache: IndicatorCache | None = None,
    fill_instrument_spec: Any = None,
) -> tuple[float, float, float]:
    ctx = IntradayStrategyContext(
        df,
        end_index=index,
        timeframe=str(dataset_validation["timeframe"]),
        bar_minutes=int(dataset_validation["bar_minutes"]),
        decision_time=call_time,
        previous_position=previous_signal,
        position_quantity=previous_contracts,
        order_statuses=execution_book.order_statuses() if execution_book is not None else [],
        indicator_cache=indicator_cache,
        equity_cash=decision_equity_cash,
        instrument_spec=instrument_spec,
        base_currency=config.portfolio.base_currency,
        available_funds=(decision_cash or {}).get(config.portfolio.base_currency)
        if instrument_spec.instrument_type == "spot"
        else None,
        point_value=(
            contract_point_value(instrument_spec)
            * native_cash_in_base(1.0, instrument_spec, config.portfolio)
        ),
    )
    decision = generate_intraday_event_plugin_decision_from_callable(
        generate_signal,
        ctx,
        config.strategy.strategy_params,
    )
    if decision.skip_unaffordable_add:
        interval = execution_interval(
            rows, index, config.execution.timing, config.execution.data_mode
        )
        if skip_unaffordable_add(
            decision,
            previous_contracts=previous_contracts,
            equity_cash=equity_cash,
            fill_price=fill_price,
            fill_row=rows[interval.fill_index],
            instrument=fill_instrument_spec or instrument_spec,
            config=config,
            decision_state=decision_state,
        ):
            return fill_price, previous_signal, previous_contracts
    sizing_price = float(row["close"])
    requested_contracts = decision_contracts(
        decision,
        sizing_price=sizing_price,
        instrument_spec=instrument_spec,
        config=config,
        equity_cash=decision_equity_cash,
    )
    if (
        config.portfolio.risk_policy == "clip"
        and decision.order_type == "market"
        and decision.time_in_force == "ioc"
        and config.execution.latency_bars == 0
    ):
        # Execution sees the opening quote after the causal signal. Cap the order
        # before submitting it, since a gap can invalidate the signal-price cap.
        requested_contracts = apply_risk_controls(
            requested_contracts,
            fill_price,
            fill_instrument_spec or instrument_spec,
            config.portfolio,
            equity_cash,
        )
    decision_state.update(
        {
            "requested_signal": decision.target,
            "strategy_reason": decision.reason,
            "strategy_features": decision.features,
            "order_type": decision.order_type,
            "order_limit_price": decision.limit_price,
            "order_stop_price": decision.stop_price,
            "order_time_in_force": decision.time_in_force,
            "order_expires_after_bars": decision.expires_after_bars,
            "order_trailing_offset": decision.trailing_offset,
            "order_oco_group": decision.oco_group,
            "order_id": decision.order_id,
            "cancel_order_id": decision.cancel_order_id,
            "replace_order_id": decision.replace_order_id,
            "take_profit_price": decision.take_profit_price,
            "stop_loss_price": decision.stop_loss_price,
            "requested_contracts": requested_contracts,
        }
    )
    if config.execution.engine == "v3" and execution_book is not None:
        order_result = execution_book.submit_and_process(
            rows=rows,
            index=index,
            order_type=decision.order_type,
            limit_price=decision.limit_price,
            stop_price=decision.stop_price,
            time_in_force=decision.time_in_force,
            expires_after_bars=decision.expires_after_bars,
            trailing_offset=decision.trailing_offset,
            oco_group=decision.oco_group,
            order_id=decision.order_id,
            cancel_order_id=decision.cancel_order_id,
            replace_order_id=decision.replace_order_id,
            take_profit_price=decision.take_profit_price,
            stop_loss_price=decision.stop_loss_price,
            post_only=decision.post_only,
            reduce_only=decision.reduce_only,
            self_trade_prevention=decision.self_trade_prevention,
            previous_signal=previous_signal,
            previous_contracts=previous_contracts,
            requested_signal=decision.target,
            requested_contracts=requested_contracts,
            base_fill_price=fill_price,
        )
    else:
        order_result = resolve_order_fill(
            order_type=decision.order_type,
            limit_price=decision.limit_price,
            stop_price=decision.stop_price,
            previous_signal=previous_signal,
            requested_signal=decision.target,
            base_fill_price=fill_price,
            row=row,
            next_row=next_row,
            execution=config.execution,
        )
    decision_state["order_status"] = order_result["status"]
    decision_state["order_filled"] = bool(order_result["filled"])
    if decision_state["order_filled"]:
        return (
            float(order_result["fill_price"]),
            float(order_result.get("target_signal", decision.target)),
            float(order_result.get("target_contracts", requested_contracts)),
        )
    decision_state["signal_reason"] = "order_unfilled"
    return fill_price, previous_signal, previous_contracts


def decision_contracts(
    decision: Any,
    *,
    sizing_price: float,
    instrument_spec: Any,
    config: IntradayEventBacktestConfig,
    equity_cash: float,
) -> float:
    if decision.target == 0.0 and decision.contracts is None:
        return 0.0
    if decision.contracts is not None:
        if decision.target != 0.0 and decision.contracts <= 0.0:
            raise StrategyPluginError("non-flat strategy decision requires contracts > 0")
        requested = float(decision.target) * float(decision.contracts)
        return apply_risk_controls(
            requested,
            sizing_price,
            instrument_spec,
            config.portfolio,
            equity_cash,
        )
    if config.portfolio.sizing_mode == "strategy_contracts":
        raise StrategyPluginError("strategy decision must return contracts")
    return target_contracts(
        decision.target,
        sizing_price,
        instrument_spec,
        config.portfolio,
        equity_cash,
    )


def apply_volume_state(
    decision_state: dict[str, Any],
    volume_result: dict[str, Any],
    previous_contracts: float,
) -> tuple[float, float]:
    decision_state["requested_contract_delta"] = (
        decision_state["requested_contracts"] - previous_contracts
    )
    decision_state["filled_contract_delta"] = float(volume_result["filled_contract_delta"])
    decision_state["fill_fraction"] = float(volume_result["fill_fraction"])
    decision_state["volume_limit_contracts"] = volume_result["volume_limit_contracts"]
    decision_state["volume_limited"] = bool(volume_result["volume_limited"])
    if decision_state["order_status"] == "rejected_insufficient_margin":
        decision_state["fill_fraction"] = 0.0
    if volume_result["status"] is not None:
        decision_state["order_status"] = str(volume_result["status"])
        decision_state["order_filled"] = bool(volume_result["filled"])
        if decision_state["signal_reason"] in {None, "order_unfilled"}:
            decision_state["signal_reason"] = (
                "partial_fill" if decision_state["order_filled"] else "volume_rejected"
            )
    return float(volume_result["target_signal"]), float(volume_result["target_contracts"])
