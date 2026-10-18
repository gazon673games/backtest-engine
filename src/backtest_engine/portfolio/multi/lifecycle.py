"""Lifecycle notifications for stateful portfolio strategies."""

from __future__ import annotations

from typing import Any

from backtest_engine.strategies.lifecycle import finish_strategy, notify_strategy_fill


def start_multi_strategy(strategy: Any, params: dict[str, str]) -> None:
    lifecycle = getattr(strategy, "__strategy_lifecycle__", None)
    if lifecycle is not None:
        lifecycle.start(params)


def notify_multi_fills(strategy: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if float(row["contract_delta"]) or bool(row.get("liquidated")):
            notify_strategy_fill(strategy, row)


def finish_multi_strategy(
    strategy: Any,
    *,
    event_count: int,
    equity_cash: float,
    positions: dict[str, float],
) -> None:
    finish_strategy(
        strategy,
        {
            "event_count": event_count,
            "final_equity_cash": equity_cash,
            "final_positions": dict(positions),
        },
    )
