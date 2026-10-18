from __future__ import annotations

from collections.abc import Callable
from typing import Any

from backtest_engine.intraday.event.data import instrument_spec_for_row

EventProgressCallback = Callable[[int, int, dict[str, Any] | None], None]


def cached_instrument_spec(
    cache: dict[tuple[Any, ...], Any],
    row: dict[str, Any],
    ts: Any,
    default_spec: Any,
) -> Any:
    key = (
        ts.date(),
        row.get("active_contract"),
        row.get("source_contract_symbol"),
        row.get("continuous_symbol"),
    )
    if key not in cache:
        cache[key] = instrument_spec_for_row(row, ts, default_spec)
    return cache[key]


def emit_progress(
    callback: Any,
    index: int,
    total: int,
    stride: int,
    event_row: dict[str, Any] | None,
    *,
    force: bool | None = None,
) -> None:
    if callback is None:
        return
    if force is None and event_row is not None:
        force = any(
            event_row.get(key) for key in ("strategy_called", "missing_decision", "contract_delta")
        )
    processed = index + 1
    if force or processed == total or processed % stride == 0:
        callback(processed, total, event_row)


def load_event_runtime(config):
    from backtest_engine.intraday.event.execution_v3 import IntradayExecutionBookV3
    from backtest_engine.strategies.plugins.loader import load_intraday_strategy_plugin

    generate_signal = load_intraday_strategy_plugin(
        config.strategy.strategy_file,
        require_metadata=config.strategy.strict_runtime,
        params=config.strategy.strategy_params,
    )
    execution_book = (
        IntradayExecutionBookV3(
            config.execution, allow_fractional_contracts=config.portfolio.allow_fractional_contracts
        )
        if config.execution.engine == "v3"
        else None
    )
    return generate_signal, execution_book
