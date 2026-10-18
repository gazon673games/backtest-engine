from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.javascript.context import JsContextEncoder
from backtest_engine.strategies.javascript.runtime import JsIntradayStrategy


def history_frame(size: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 2) + timedelta(minutes=index) for index in range(size)],
            "timeframe": ["1m"] * size,
            "close": [float(index) for index in range(size)],
        }
    )


def test_js_context_encoder_sends_only_new_bars_after_initial_sync() -> None:
    frame = history_frame(5_002)
    encoder = JsContextEncoder()

    initial = encoder.encode(
        IntradayStrategyContext(frame, end_index=4_999),
        history_bars=5_000,
        data_columns=("ts", "close"),
    )
    delta = encoder.encode(
        IntradayStrategyContext(frame, end_index=5_000),
        history_bars=5_000,
        data_columns=("ts", "close"),
    )

    assert initial["historySync"]["reset"] is True
    assert initial["historySync"]["data"]["length"] == 5_000
    assert delta["historySync"]["reset"] is False
    assert delta["historySync"]["data"]["length"] == 1
    assert delta["sessionSync"]["data"]["length"] == 1


def test_js_bridge_reconstructs_history_from_deltas(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for JS strategy runtime tests")
    plugin_path = tmp_path / "history.js"
    plugin_path.write_text(
        "function onDecision(ctx) { return { target: 0, features: { "
        "length: ctx.history.length, last: ctx.history.value('close') } }; }\n",
        encoding="utf-8",
    )
    frame = history_frame(5_002)
    strategy = JsIntradayStrategy(plugin_path, history_bars=5_000)
    try:
        first = strategy.execute(IntradayStrategyContext(frame, end_index=4_999), {})
        second = strategy.execute(IntradayStrategyContext(frame, end_index=5_000), {})
    finally:
        strategy.close()

    assert first["features"] == {"length": 5_000, "last": 4_999.0}
    assert second["features"] == {"length": 5_000, "last": 5_000.0}


def test_js_bridge_exposes_explicit_lifecycle_state(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for JS strategy runtime tests")
    plugin_path = tmp_path / "lifecycle.js"
    plugin_path.write_text(
        "function initialize() { return { calls: 0, fills: 0 }; }\n"
        "function onFill(state) { state.fills += 1; }\n"
        "function onDecision(ctx) { ctx.state.calls += 1; return { target: 0, "
        "features: { calls: ctx.state.calls, fills: ctx.state.fills } }; }\n",
        encoding="utf-8",
    )
    frame = history_frame(2)
    strategy = JsIntradayStrategy(plugin_path)
    try:
        first = strategy.execute(IntradayStrategyContext(frame, end_index=0), {})
        strategy.notify_fill({"quantity": 1})
        second = strategy.execute(IntradayStrategyContext(frame, end_index=1), {})
        strategy.finish({"event_count": 2})
    finally:
        strategy.close()

    assert first["features"] == {"calls": 1, "fills": 0}
    assert second["features"] == {"calls": 2, "fills": 1}


def test_js_value_selection_filters_before_taking_tail_and_preserves_source(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for JS strategy runtime tests")
    plugin_path = tmp_path / "selection.js"
    plugin_path.write_text(
        "function onDecision(ctx) {\n"
        "  const copy = ctx.history.values('close'); copy[0] = 999;\n"
        "  return { target: 0, features: {\n"
        "    bounded: ctx.history.values('close', { from: '00:01', to: '00:04' }),\n"
        "    tail: ctx.history.values('close', { from: '00:01', to: '00:04', bars: 2 }),\n"
        "    alias: ctx.history.values('close', { fromBarsAgo: 2 }),\n"
        "    empty: ctx.history.values('close', { from: '01:00' }),\n"
        "    first: ctx.history.value('close', 0)\n"
        "  } };\n"
        "}\n",
        encoding="utf-8",
    )
    strategy = JsIntradayStrategy(plugin_path)
    try:
        result = strategy.execute(IntradayStrategyContext(history_frame(6), end_index=5), {})
    finally:
        strategy.close()

    assert result["features"] == {
        "bounded": [1.0, 2.0, 3.0, 4.0],
        "tail": [3.0, 4.0],
        "alias": [4.0, 5.0],
        "empty": [],
        "first": 0.0,
    }
