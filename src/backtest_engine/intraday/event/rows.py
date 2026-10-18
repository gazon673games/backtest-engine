from __future__ import annotations

from datetime import datetime, time
from typing import Any

from backtest_engine.accounting.crypto import CryptoSettlement
from backtest_engine.data.bar_availability import row_available_at, row_open_ts
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig
from backtest_engine.intraday.event.futures import event_cash_currency
from backtest_engine.intraday.event.schedule import session_label_for_event
from backtest_engine.intraday.event.utils import (
    decision_time_text_from_value,
    decision_times_text,
    json_dumps,
)


def build_event_row(
    *,
    config: IntradayEventBacktestConfig,
    row: dict[str, Any],
    exit_row: dict[str, Any],
    current_ts: datetime,
    fill_ts: datetime,
    fill_price: float,
    exit_price: float,
    call_time: time | None,
    missing_decision_times: list[time],
    decision_state: dict[str, Any],
    position_state: dict[str, float],
    instrument_state: dict[str, Any],
    pnl_state: dict[str, float],
    cost_parts: dict[str, float],
    contract_roll: bool,
    crypto_settlement: CryptoSettlement | None = None,
) -> dict[str, Any]:
    instrument = instrument_state["instrument_spec"]
    return {
        "ts": current_ts,
        "next_ts": row_available_at(exit_row),
        "fill_ts": fill_ts,
        "bar_open_ts": row_open_ts(row),
        "bar_close_ts": row_available_at(row),
        "close": position_state["close"],
        "next_close": float(exit_row["close"]),
        "fill_price": fill_price,
        "exit_price": exit_price,
        "active_contract": str(row["active_contract"]),
        "next_active_contract": str(exit_row["active_contract"]),
        "source_contract_symbol": str(row["source_contract_symbol"]),
        "next_source_contract_symbol": str(exit_row["source_contract_symbol"]),
        "contract_roll": contract_roll,
        "roll_pnl_policy": config.execution.roll_pnl_policy,
        "roll_pnl_excluded": bool(
            contract_roll
            and (
                position_state["current_signal"] != 0.0
                or position_state["current_contracts"] != 0.0
            )
            and config.execution.roll_pnl_policy == "exclude_gap"
        ),
        "session_label": session_label_for_event(row_open_ts(row), config.session),
        "next_session_label": session_label_for_event(row_open_ts(exit_row), config.session),
        "decision_time": decision_time_text_from_value(call_time),
        "strategy_called": decision_state["strategy_called"],
        "strategy_reason": decision_state["strategy_reason"],
        "strategy_features": json_dumps(decision_state["strategy_features"]),
        "requested_position": decision_state["requested_signal"],
        "requested_contracts": decision_state["requested_contracts"],
        "requested_contract_delta": decision_state["requested_contract_delta"],
        "order_type": decision_state["order_type"],
        "order_status": decision_state["order_status"],
        "order_filled": decision_state["order_filled"],
        "order_limit_price": decision_state["order_limit_price"],
        "order_stop_price": decision_state["order_stop_price"],
        "order_time_in_force": decision_state["order_time_in_force"],
        "order_id": decision_state["order_id"],
        "cancel_order_id": decision_state["cancel_order_id"],
        "replace_order_id": decision_state["replace_order_id"],
        "take_profit_price": decision_state["take_profit_price"],
        "stop_loss_price": decision_state["stop_loss_price"],
        "filled_contract_delta": decision_state["filled_contract_delta"],
        "fill_fraction": decision_state["fill_fraction"],
        "volume_limit_contracts": decision_state["volume_limit_contracts"],
        "volume_limited": decision_state["volume_limited"],
        "missing_decision": decision_state["missing_decision"],
        "missing_decision_policy": decision_state["missing_decision_policy"],
        "missing_decision_time": decision_times_text(missing_decision_times),
        "target_position": position_state["current_signal"],
        "previous_position": position_state["previous_signal"],
        "target_contracts": position_state["current_contracts"],
        "previous_contracts": position_state["previous_contracts"],
        "contract_delta": position_state["contract_delta"],
        "asset_class": instrument.asset_class,
        "instrument_type": instrument.instrument_type,
        "instrument_symbol": instrument.symbol,
        "price_currency": instrument.price_currency,
        "pnl_currency": event_cash_currency(instrument, config.portfolio),
        "contract_multiplier": instrument.contract_multiplier,
        "lot_size": instrument.lot_size,
        "tick_size": instrument.tick_size,
        "tick_value": (
            None
            if instrument.tick_value is None
            else instrument.tick_value * instrument_state["cash_to_base_rate"]
        ),
        "initial_margin": instrument_state["initial_margin"],
        "exchange_fee_per_contract": (
            instrument.exchange_fee_per_contract * instrument_state["cash_to_base_rate"]
        ),
        "point_value": instrument_state["point_value"],
        "position_notional": instrument_state["position_notional"],
        "position_turnover": position_state["turnover"],
        "roll_turnover": (2.0 * abs(position_state["current_signal"]) if contract_roll else 0.0),
        "roll_contract_turnover": cost_parts["roll_contract_turnover"],
        "turnover": position_state["turnover"]
        + (2.0 * abs(position_state["current_signal"]) if contract_roll else 0.0),
        "position_turnover_notional": instrument_state["turnover_notional"],
        "roll_turnover_notional": cost_parts["roll_turnover_notional"],
        "turnover_notional": (
            instrument_state["turnover_notional"] + cost_parts["roll_turnover_notional"]
        ),
        "asset_return": pnl_state["asset_return"],
        "gross_return": pnl_state["gross_return"],
        "gross_pnl": pnl_state["gross_pnl"],
        "cost": pnl_state["cost"],
        "cost_cash": pnl_state["cost_cash"],
        "commission_cost_cash": cost_parts["commission_cost_cash"],
        "slippage_cost_cash": cost_parts["slippage_cost_cash"],
        "spread_cost_cash": cost_parts["spread_cost_cash"],
        "roll_cost_cash": cost_parts["roll_cost_cash"],
        "trading_fee_cash": cost_parts["trading_fee_cash"],
        "funding_cost_cash": cost_parts["funding_cost_cash"],
        "liquidation_fee_cash": cost_parts["liquidation_fee_cash"],
        "maintenance_margin": settlement_value(
            crypto_settlement,
            "maintenance_margin",
            instrument_state["cash_to_base_rate"],
        ),
        "liquidated": bool(crypto_settlement and crypto_settlement.liquidated),
        "liquidation_price": (
            None if crypto_settlement is None else crypto_settlement.liquidation_price
        ),
        "bankruptcy_price": (
            None if crypto_settlement is None else crypto_settlement.bankruptcy_price
        ),
        "liquidation_quantity": (
            0.0 if crypto_settlement is None else crypto_settlement.liquidation_quantity
        ),
        "remaining_quantity": (
            position_state["current_contracts"]
            if crypto_settlement is None
            else crypto_settlement.remaining_quantity
        ),
        "liquidation_status": (
            "none" if crypto_settlement is None else crypto_settlement.liquidation_status
        ),
        "net_return": pnl_state["net_return"],
        "net_pnl": pnl_state["net_pnl"],
        "equity": pnl_state["current_equity"],
        "equity_cash": pnl_state["current_equity_cash"],
        "execution_timing": config.execution.timing,
        "sizing_mode": config.portfolio.sizing_mode,
        "signal_reason": decision_state["signal_reason"],
    }


def settlement_value(
    settlement: CryptoSettlement | None,
    field: str,
    cash_to_base_rate: float,
) -> float:
    if settlement is None:
        return 0.0
    return float(getattr(settlement, field)) * cash_to_base_rate
