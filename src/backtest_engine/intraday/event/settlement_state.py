from __future__ import annotations

from typing import Any


def apply_settlement_state(
    result: Any,
    *,
    current_signal: float,
    current_contracts: float,
    previous_contracts: float,
    position_state: dict[str, float],
    instrument_state: dict[str, Any],
) -> tuple[float, float]:
    if result.ending_quantity == current_contracts:
        return current_signal, current_contracts
    original_contracts = current_contracts
    current_contracts = result.ending_quantity
    current_signal = (
        0.0
        if original_contracts == 0.0
        else current_signal * current_contracts / original_contracts
    )
    position_state["current_signal"] = current_signal
    position_state["current_contracts"] = current_contracts
    position_state["contract_delta"] = current_contracts - previous_contracts
    instrument_state["position_notional"] = (
        abs(current_contracts) * result.effective_exit_price * instrument_state["point_value"]
    )
    return current_signal, current_contracts
