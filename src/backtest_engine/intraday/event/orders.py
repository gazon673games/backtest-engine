from __future__ import annotations

import math
from typing import Any

from backtest_engine.execution.model import (
    event_loop_stop as execution_model_event_loop_stop,
)
from backtest_engine.execution.model import (
    execution_interval_volume as model_execution_interval_volume,
)
from backtest_engine.execution.model import (
    resolve_order_fill as model_resolve_order_fill,
)
from backtest_engine.intraday.event.config import IntradayEventExecutionConfig


def event_loop_stop(height: int, execution: IntradayEventExecutionConfig) -> int:
    return execution_model_event_loop_stop(height, execution.timing)


def resolve_order_fill(
    *,
    order_type: str,
    limit_price: float | None,
    stop_price: float | None,
    previous_signal: float,
    requested_signal: float,
    base_fill_price: float,
    row: dict[str, Any],
    next_row: dict[str, Any],
    execution: IntradayEventExecutionConfig,
) -> dict[str, Any]:
    fill = model_resolve_order_fill(
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        previous_signal=previous_signal,
        requested_signal=requested_signal,
        base_fill_price=base_fill_price,
        row=row,
        next_row=next_row,
        timing=execution.timing,
        intrabar_ambiguity_policy=execution.intrabar_ambiguity_policy,
    )
    return {"filled": fill.filled, "status": fill.status, "fill_price": fill.fill_price}


def apply_volume_participation_limit(
    *,
    previous_signal: float,
    previous_contracts: float,
    target_signal: float,
    target_contracts: float,
    row: dict[str, Any],
    next_row: dict[str, Any],
    execution: IntradayEventExecutionConfig,
    allow_fractional_contracts: bool = True,
) -> dict[str, Any]:
    requested_delta = target_contracts - previous_contracts
    if execution.volume_participation_limit is None or requested_delta == 0.0:
        return volume_result(
            target_signal,
            target_contracts,
            requested_delta,
            1.0,
            None,
            False,
            None,
            True,
        )
    volume_limit_contracts = (
        execution_interval_volume(row, next_row, execution)
        * execution.volume_participation_limit
        * execution.volume_to_contracts_multiplier
    )
    if not allow_fractional_contracts:
        volume_limit_contracts = float(math.floor(volume_limit_contracts))
    return limited_volume_result(
        previous_signal=previous_signal,
        previous_contracts=previous_contracts,
        target_signal=target_signal,
        target_contracts=target_contracts,
        requested_delta=requested_delta,
        volume_limit_contracts=volume_limit_contracts,
        execution=execution,
    )


def confirmed_volume_result(
    *,
    previous_contracts: float,
    target_signal: float,
    target_contracts: float,
    requested_contracts: float,
) -> dict[str, Any]:
    requested_delta = requested_contracts - previous_contracts
    filled_delta = target_contracts - previous_contracts
    fraction = abs(filled_delta / requested_delta) if requested_delta else 1.0
    return volume_result(
        target_signal,
        target_contracts,
        filled_delta,
        fraction,
        None,
        fraction < 1.0,
        None,
        filled_delta != 0.0,
    )


def limited_volume_result(
    *,
    previous_signal: float,
    previous_contracts: float,
    target_signal: float,
    target_contracts: float,
    requested_delta: float,
    volume_limit_contracts: float,
    execution: IntradayEventExecutionConfig,
) -> dict[str, Any]:
    abs_requested_delta = abs(requested_delta)
    if volume_limit_contracts >= abs_requested_delta:
        return volume_result(
            target_signal,
            target_contracts,
            requested_delta,
            1.0,
            volume_limit_contracts,
            False,
            None,
            True,
        )
    if execution.partial_fill_policy == "reject" or volume_limit_contracts <= 0.0:
        return volume_result(
            previous_signal,
            previous_contracts,
            0.0,
            0.0,
            volume_limit_contracts,
            True,
            "volume_rejected",
            False,
        )
    filled_contract_delta = math.copysign(volume_limit_contracts, requested_delta)
    fill_fraction = abs(filled_contract_delta) / abs_requested_delta
    return volume_result(
        previous_signal + (target_signal - previous_signal) * fill_fraction,
        previous_contracts + filled_contract_delta,
        filled_contract_delta,
        fill_fraction,
        volume_limit_contracts,
        True,
        "partial_filled",
        True,
    )


def volume_result(
    target_signal: float,
    target_contracts: float,
    filled_contract_delta: float,
    fill_fraction: float,
    volume_limit_contracts: float | None,
    volume_limited: bool,
    status: str | None,
    filled: bool,
) -> dict[str, Any]:
    return {
        "target_signal": target_signal,
        "target_contracts": target_contracts,
        "filled_contract_delta": filled_contract_delta,
        "fill_fraction": fill_fraction,
        "volume_limit_contracts": volume_limit_contracts,
        "volume_limited": volume_limited,
        "status": status,
        "filled": filled,
    }


def execution_interval_volume(
    row: dict[str, Any],
    next_row: dict[str, Any],
    execution: IntradayEventExecutionConfig,
) -> float:
    return model_execution_interval_volume(row, next_row, execution.timing, execution.data_mode)


def resolved_volume_result(
    *,
    execution: IntradayEventExecutionConfig,
    allow_fractional_contracts: bool,
    confirmed: bool,
    previous_signal: float,
    previous_contracts: float,
    target_signal: float,
    target_contracts: float,
    requested_contracts: float,
    row: dict[str, Any],
    next_row: dict[str, Any],
) -> dict[str, Any]:
    if not confirmed:
        return apply_volume_participation_limit(
            previous_signal=previous_signal,
            previous_contracts=previous_contracts,
            target_signal=target_signal,
            target_contracts=target_contracts,
            row=row,
            next_row=next_row,
            execution=execution,
            allow_fractional_contracts=allow_fractional_contracts,
        )
    result = confirmed_volume_result(
        previous_contracts=previous_contracts,
        target_signal=target_signal,
        target_contracts=target_contracts,
        requested_contracts=requested_contracts,
    )
    limit = execution.volume_participation_limit
    if limit is not None:
        capacity = (
            execution_interval_volume(row, next_row, execution)
            * limit
            * execution.volume_to_contracts_multiplier
        )
        result["volume_limit_contracts"] = (
            capacity if allow_fractional_contracts else math.floor(capacity)
        )
        if not result["filled"] and abs(requested_contracts - previous_contracts) > capacity:
            result["status"] = "volume_rejected"
            result["volume_limited"] = True
    return result
