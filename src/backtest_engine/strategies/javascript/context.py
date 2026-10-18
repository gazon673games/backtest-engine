from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

import polars as pl

from backtest_engine.strategies.intraday_api import IntradayStrategyContext

DEFAULT_HISTORY_BARS = 5000


@dataclass
class _SyncState:
    source_token: int
    end_index: int
    current_date: date
    history_bars: int
    data_columns: tuple[str, ...]


class JsContextEncoder:
    """Encode only bars added since the preceding decision."""

    def __init__(self) -> None:
        self._state: _SyncState | None = None

    def reset(self) -> None:
        self._state = None

    def encode(
        self,
        ctx: IntradayStrategyContext,
        *,
        history_bars: int,
        data_columns: tuple[str, ...],
    ) -> dict[str, Any]:
        previous = self._state
        reset = (
            previous is None
            or previous.source_token != ctx.history_source_token
            or previous.end_index >= ctx.end_index
            or previous.history_bars != history_bars
            or previous.data_columns != data_columns
        )
        added_bars = history_bars if reset else ctx.end_index - previous.end_index
        history_reset = reset or added_bars > history_bars
        history = ctx.history_tail(history_bars if history_reset else added_bars)
        session_reset = reset or previous.current_date != ctx.current_date
        session = ctx.session_history() if session_reset else ctx.history_tail(added_bars)
        self._state = _SyncState(
            source_token=ctx.history_source_token,
            end_index=ctx.end_index,
            current_date=ctx.current_date,
            history_bars=history_bars,
            data_columns=data_columns,
        )
        return {
            **js_context_scalars(ctx),
            "historySync": {
                "reset": history_reset,
                "limit": history_bars,
                "data": frame_payload(history, data_columns=data_columns),
            },
            "sessionSync": {
                "reset": session_reset,
                "data": frame_payload(session, data_columns=data_columns),
            },
        }


def js_context_payload(
    ctx: IntradayStrategyContext,
    *,
    history_bars: int = DEFAULT_HISTORY_BARS,
    data_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        **js_context_scalars(ctx),
        "history": frame_payload(ctx.history_tail(history_bars), data_columns=data_columns),
        "session": frame_payload(ctx.session_history(), data_columns=data_columns),
    }


def js_context_scalars(ctx: IntradayStrategyContext) -> dict[str, Any]:
    current_ts = ctx.current_ts
    return {
        "bar": scalar_mapping(ctx.current_bar()),
        "timeframe": ctx.timeframe,
        "barMinutes": ctx.bar_minutes,
        "decisionTime": None
        if ctx.decision_time is None
        else ctx.decision_time.isoformat(timespec="minutes"),
        "currentTs": current_ts.isoformat(),
        "date": ctx.current_date.isoformat(),
        "time": ctx.current_time.isoformat(timespec="minutes"),
        "weekday": ctx.weekday,
        "position": ctx.previous_position,
        "previousPosition": ctx.previous_position,
        "activeContract": ctx.active_contract,
        "equityCash": ctx.equity_cash,
        "pointValue": ctx.point_value,
        "orderStatuses": [scalar_mapping(item) for item in ctx.order_statuses()],
    }


def frame_payload(
    frame: pl.DataFrame,
    *,
    data_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    requested = data_columns or tuple(frame.columns)
    columns = [name for name in requested if name in frame.columns]
    return {
        "length": frame.height,
        "columns": {
            name: [json_default(value) for value in frame.get_column(name).to_list()]
            for name in columns
        },
    }


def scalar_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    return {str(key): json_default(value) for key, value in raw.items()}


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date | time):
        return value.isoformat()
    return value
