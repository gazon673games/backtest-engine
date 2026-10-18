from __future__ import annotations

import math
from typing import Any

import polars as pl

from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig
from backtest_engine.intraday.event.output_buffer import EventFrameBuffer
from backtest_engine.intraday.event.risk_safety import validate_event_end
from backtest_engine.strategies.lifecycle import finish_strategy, notify_strategy_fill


def notify_event_fill(strategy: Any, event: dict[str, Any]) -> None:
    if bool(event["liquidated"]):
        # A candle may contain both an order fill and a later margin reduction.
        # Deliver both confirmed transitions, preserving the partial residual.
        order_delta = float(event["filled_contract_delta"])
        before_liquidation = float(event["previous_contracts"]) + order_delta
        if order_delta:
            notify_strategy_fill(
                strategy,
                {
                    **event,
                    "contract_delta": order_delta,
                    "target_contracts": before_liquidation,
                    "liquidated": False,
                    "fill_id": str((event.get("order_id"), event["fill_ts"], "order")),
                },
            )
        remaining = float(event["remaining_quantity"])
        notify_strategy_fill(
            strategy,
            {
                **event,
                "previous_contracts": before_liquidation,
                "contract_delta": -math.copysign(event["liquidation_quantity"], before_liquidation),
                "target_contracts": remaining,
                "fill_price": event["liquidation_price"],
                "fill_ts": event["next_ts"],
                "liquidated": remaining == 0,
                "fill_id": str((event.get("order_id"), event["next_ts"], "liquidation")),
            },
        )
        return
    if float(event["contract_delta"]) or bool(event["liquidated"]):
        notify_strategy_fill(strategy, event)


def finalize_event_run(
    strategy: Any,
    buffer: EventFrameBuffer,
    ledger: BacktestLedgerRuntime,
    snapshots: list[dict[str, Any]] | None,
    signal: float,
    quantity: float,
    *,
    config: IntradayEventBacktestConfig,
) -> pl.DataFrame:
    validate_event_end(config, quantity)
    if snapshots is not None:
        snapshots.extend(ledger.snapshots)
    events = buffer.finish()
    finish_event_strategy(strategy, events, ledger.equity_cash, signal, quantity)
    return events


def finish_event_strategy(
    strategy: Any,
    events: pl.DataFrame,
    equity_cash: float,
    signal: float,
    quantity: float,
) -> None:
    finish_strategy(
        strategy,
        {
            "event_count": events.height,
            "final_equity_cash": equity_cash,
            "final_position": signal,
            "final_quantity": quantity,
        },
    )
