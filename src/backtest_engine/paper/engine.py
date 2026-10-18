"""Paper runner using the production strategy, execution, and ledger contracts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.bar_availability import row_available_at
from backtest_engine.execution.models import FillEvent, OrderRequest, OrderState
from backtest_engine.execution.simulator import ExecutionSimulatorV3
from backtest_engine.paper.accounting import PaperAccounting, as_datetime
from backtest_engine.paper.models import PaperTradingConfig, PaperTradingResult
from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.lifecycle import finish_strategy, notify_strategy_fill
from backtest_engine.strategies.plugins.loader import (
    generate_intraday_event_plugin_decision_from_callable,
    load_intraday_strategy_plugin,
)


class PaperTradingEngine:
    def __init__(self, config: PaperTradingConfig) -> None:
        self.config = config
        self.strategy = load_intraday_strategy_plugin(
            config.strategy_file,
            params=config.strategy_params,
            require_metadata=config.strict_runtime,
        )
        metadata = getattr(self.strategy, "__strategy_metadata__", None)
        if metadata is not None and config.max_history_bars < max(
            metadata.history_bars, metadata.warmup_bars
        ):
            raise ValueError("max_history_bars is smaller than strategy history/warmup requirement")
        self.simulator = ExecutionSimulatorV3(
            volume_participation_limit=config.volume_participation_limit,
            latency_bars=config.latency_events,
            queue_ahead_quantity=config.queue_ahead_quantity,
            market_impact_bps=config.market_impact_bps,
            data_mode=config.data_mode,
        )
        self.accounting = PaperAccounting(
            instrument=config.instrument,
            initial_capital=config.initial_capital,
            base_currency=config.base_currency,
            collateral_currency=config.collateral_currency,
            leverage=config.leverage,
            liquidity_role=config.liquidity_role,
            require_mark_price=config.require_mark_price,
            require_margin_tiers=config.require_margin_tiers,
            funding_mode=config.funding_mode,
        )
        self.history: list[dict[str, Any]] = []
        self.state_rows: list[dict[str, Any]] = []
        self.position = 0.0
        self.fill_count = 0
        self.sequence = 0
        self.event_count = 0
        self.last_available_at: datetime | None = None
        self.order_targets: dict[str, float] = {}
        self.feedback_ids: set[str] = set()

    def on_event(self, raw_row: dict[str, Any]) -> dict[str, Any]:
        row = self._validated_row(raw_row)
        index = self.event_count
        self.event_count += 1
        fills = self.simulator.process_bar(row, bar_index=index)
        self._apply_fills(fills)
        self.history.append(row)
        del self.history[: -self.config.max_history_bars]
        accounting = self.accounting.settle_event(row, fills)
        if self.accounting.perpetual is not None:
            self.position = self.accounting.perpetual.quantity
        decision = generate_intraday_event_plugin_decision_from_callable(
            self.strategy,
            self._strategy_context(),
            self.config.strategy_params,
        )
        requested = (
            decision.target * decision.contracts
            if decision.contracts is not None
            else decision.target * self.config.fixed_contracts
        )
        self._apply_order_controls(decision.cancel_order_id, decision.replace_order_id, index)
        self._submit_target(decision, requested, row, index)
        state = {
            "ts": row["ts"],
            "available_at": row_available_at(row),
            "position": self.position,
            "requested_position": requested,
            "equity_cash": accounting["equity_cash"],
            "cost_cash": accounting["cost_cash"],
            "fill_count": len(fills),
            "open_orders": len(self.simulator.open_order_ids),
            "liquidated": accounting["liquidated"],
        }
        self.state_rows.append(state)
        del self.state_rows[: -self.config.max_state_rows]
        if self.event_count % self.config.checkpoint_every_events == 0:
            self._checkpoint()
        return state

    def finish(self) -> PaperTradingResult:
        result = PaperTradingResult(
            event_count=self.event_count,
            fill_count=self.fill_count,
            final_position=self.position,
            final_equity_cash=self.accounting.equity_cash,
            state_rows=self.state_rows,
            checkpoint_path=self.config.checkpoint_path,
        )
        finish_strategy(self.strategy, result.model_dump(mode="json"))
        self._checkpoint(final=True)
        return result

    def _validated_row(self, raw: dict[str, Any]) -> dict[str, Any]:
        row = dict(raw)
        row.setdefault("timeframe", "1m")
        ts = as_datetime(row["ts"])
        row["ts"] = ts
        available = row_available_at(row)
        available = as_datetime(available)
        if self.last_available_at is not None and available <= self.last_available_at:
            raise DatasetValidationError(
                "Paper events must have strictly increasing point-in-time availability"
            )
        self.last_available_at = available
        return row

    def _strategy_context(self) -> IntradayStrategyContext:
        frame = pl.DataFrame(self.history, infer_schema_length=None, strict=False)
        return IntradayStrategyContext(
            frame,
            previous_position=self.position,
            position_quantity=self.position,
            order_statuses=[
                self._order_status(self.simulator.orders[key])
                for key in self.simulator.open_order_ids | self.feedback_ids
            ],
            equity_cash=self.accounting.equity_cash,
            instrument_spec=self.config.instrument,
            base_currency=self.config.base_currency,
            available_funds=self.accounting.ledger.ledger.cash.get(self.config.base_currency, 0.0),
            point_value=self.config.instrument.contract_multiplier
            * self.config.instrument.lot_size,
        )

    def _apply_fills(self, fills: list[FillEvent]) -> None:
        self.feedback_ids = {fill.order_id for fill in fills}
        for index, fill in enumerate(fills):
            adjusted = replace(fill, liquidity_role=self.config.liquidity_role)
            fills[index] = adjusted
            self.position += adjusted.quantity if adjusted.side == "buy" else -adjusted.quantity
            self.fill_count += 1
            notify_strategy_fill(
                self.strategy,
                {
                    "order_id": adjusted.order_id,
                    "ts": adjusted.ts,
                    "side": adjusted.side,
                    "quantity": adjusted.quantity,
                    "price": adjusted.price,
                    "position": self.position,
                },
            )

    def _apply_order_controls(
        self, cancel_order_id: str | None, replace_order_id: str | None, index: int
    ) -> None:
        for order_id in {cancel_order_id, replace_order_id} - {None}:
            if order_id in self.simulator.orders:
                self.simulator.cancel(str(order_id), bar_index=index)

    def _submit_target(self, decision: Any, target: float, row: dict[str, Any], index: int) -> None:
        active = self.simulator.active_orders()
        same_target = all(
            self.order_targets.get(item.request.order_id) == target for item in active
        )
        if active and same_target:
            return
        for state in active:
            self.simulator.cancel(state.request.order_id, bar_index=index, reason="replaced")
        delta = target - self.position
        if abs(delta) <= 1e-12:
            return
        self.sequence += 1
        order_id = decision.order_id or f"paper-{self.sequence}"
        request = OrderRequest(
            order_id=order_id,
            side="buy" if delta > 0.0 else "sell",
            quantity=abs(delta),
            order_type=decision.order_type,
            limit_price=decision.limit_price,
            stop_price=decision.stop_price,
            time_in_force=decision.time_in_force,
            expires_after_bars=decision.expires_after_bars,
            trailing_offset=decision.trailing_offset,
            oco_group=decision.oco_group,
            post_only=decision.post_only,
            reduce_only=decision.reduce_only,
            self_trade_prevention=decision.self_trade_prevention,
        )
        self.simulator.submit(
            request,
            bar_index=index,
            created_ts=row["ts"],
            position_quantity=self.position,
        )
        self.order_targets[order_id] = target

    def _checkpoint(self, *, final: bool = False) -> None:
        path = self.config.checkpoint_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "final": final,
            "event_count": self.event_count,
            "fill_count": self.fill_count,
            "position": self.position,
            "equity_cash": self.accounting.equity_cash,
            "last_available_at": self.last_available_at,
            "open_orders": [self._order_status(item) for item in self.simulator.active_orders()],
            "strategy_file": str(self.config.strategy_file),
        }
        temporary = Path(f"{path}.tmp")
        temporary.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _order_status(state: OrderState) -> dict[str, Any]:
        return {
            "order_id": state.request.order_id,
            "status": state.status,
            "side": state.request.side,
            "quantity": state.request.quantity,
            "filled_quantity": state.filled_quantity,
            "remaining_quantity": state.remaining_quantity,
            "avg_fill_price": state.avg_fill_price,
            "rejection_reason": state.rejection_reason,
        }


def run_paper_feed(
    config: PaperTradingConfig,
    feed: Iterable[dict[str, Any]],
    *,
    max_events: int | None = None,
) -> PaperTradingResult:
    engine = PaperTradingEngine(config)
    events = feed if max_events is None else islice(feed, max(max_events, 0))
    for row in events:
        engine.on_event(row)
    return engine.finish()
