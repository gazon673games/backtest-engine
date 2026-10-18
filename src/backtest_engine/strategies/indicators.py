from __future__ import annotations

import math
from datetime import datetime, time
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.strategies.resampling import aggregate_ohlcv


def visible_history(ctx: Any) -> pl.DataFrame:
    return ctx.history()


def current_close(ctx: Any, column: str = "close") -> float:
    return _float_value(ctx.current_bar().get(column), column)


def sma(ctx: Any, column: str, window: int) -> float | None:
    return ctx.sma(column, window)


def ema(ctx: Any, column: str, window: int) -> float | None:
    return ctx.ema(column, window)


def rolling_high(ctx: Any, column: str = "high", window: int = 1) -> float | None:
    values = _numeric_tail(ctx, column, window, "rolling_high")
    return None if values is None else max(values)


def rolling_low(ctx: Any, column: str = "low", window: int = 1) -> float | None:
    values = _numeric_tail(ctx, column, window, "rolling_low")
    return None if values is None else min(values)


def rolling_mean(ctx: Any, column: str, window: int) -> float | None:
    values = _numeric_tail(ctx, column, window, "rolling_mean")
    return None if values is None else sum(values) / len(values)


def rolling_std(ctx: Any, column: str, window: int) -> float | None:
    values = _numeric_tail(ctx, column, window, "rolling_std")
    if values is None:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def zscore(ctx: Any, column: str, window: int) -> float | None:
    mean = rolling_mean(ctx, column, window)
    std = rolling_std(ctx, column, window)
    if mean is None or std is None or std == 0.0:
        return None
    return (current_close(ctx, column) - mean) / std


def resample_ohlcv(ctx: Any, every: str) -> pl.DataFrame:
    resampler = getattr(ctx, "resample_ohlcv", None)
    if resampler is not None:
        return resampler(every)
    return aggregate_ohlcv(ctx.history(), every).drop_nulls("close")


def higher_timeframe_sma(
    ctx: Any,
    every: str,
    column: str = "close",
    window: int = 1,
) -> float | None:
    _validate_window(window, "higher_timeframe_sma")
    frame = resample_ohlcv(ctx, every)
    if column not in frame.columns:
        raise StrategyPluginError(f"higher_timeframe_sma requested missing column: {column}")
    if frame.height < window:
        return None
    values = frame.get_column(column).tail(window)
    if values.null_count() > 0:
        return None
    return float(values.mean())


def atr(ctx: Any, window: int) -> float | None:
    _validate_window(window, "atr")
    history = ctx.history()
    if history.height < window + 1:
        return None
    required = {"high", "low", "close"}
    missing = required.difference(history.columns)
    if missing:
        raise StrategyPluginError(f"ATR requires columns: {', '.join(sorted(missing))}")
    rows = history.tail(window + 1).iter_rows(named=True)
    previous_close = None
    true_ranges: list[float] = []
    for row in rows:
        high = _float_value(row["high"], "high")
        low = _float_value(row["low"], "low")
        close = _float_value(row["close"], "close")
        if previous_close is not None:
            true_ranges.append(
                max(high - low, abs(high - previous_close), abs(low - previous_close))
            )
        previous_close = close
    if len(true_ranges) < window:
        return None
    return sum(true_ranges) / len(true_ranges)


def rsi(ctx: Any, column: str = "close", window: int = 14) -> float | None:
    _validate_window(window, "rsi")
    values = _numeric_tail(ctx, column, window + 1, "rsi")
    if values is None:
        return None
    gains = []
    losses = []
    for previous, current in zip(values[:-1], values[1:], strict=True):
        change = current - previous
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window
    if avg_loss == 0.0:
        return 100.0
    relative_strength = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def ensure_min_bars(ctx: Any, bars: int) -> bool:
    _validate_window(bars, "ensure_min_bars")
    return ctx.history().height >= bars


def bars_since_time(ctx: Any, raw_time: str) -> int:
    target_time = _parse_time(raw_time)
    count = 0
    for value in reversed(ctx.history().get_column("ts")):
        ts = _datetime_value(value)
        if ts.date() != ctx.current_date:
            break
        if ts.time() < target_time:
            break
        count += 1
    return count


def in_time_window(ctx: Any, start: str, end: str) -> bool:
    current_time = ctx.current_time
    return _parse_time(start) <= current_time <= _parse_time(end)


def is_weekday(ctx: Any, *weekdays: int) -> bool:
    return ctx.weekday in weekdays


def _numeric_tail(ctx: Any, column: str, window: int, helper: str) -> list[float] | None:
    _validate_window(window, helper)
    series = ctx.series(column)
    if len(series) < window:
        return None
    tail = series.tail(window)
    if tail.null_count() > 0:
        return None
    return [_float_value(value, column) for value in tail.to_list()]


def _validate_window(window: int, helper: str) -> None:
    if window <= 0:
        raise StrategyPluginError(f"{helper} window must be greater than 0")


def _float_value(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise StrategyPluginError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise StrategyPluginError(f"{name} must be finite")
    return parsed


def _datetime_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    raise StrategyPluginError(f"ts must contain datetime values, got {value!r}")


def _parse_time(raw_time: str) -> time:
    try:
        parsed = time.fromisoformat(raw_time)
    except ValueError as exc:
        raise StrategyPluginError(f"Invalid time: {raw_time}. Expected HH:MM") from exc
    if parsed.second != 0 or parsed.microsecond != 0:
        raise StrategyPluginError(f"Invalid time: {raw_time}. Expected minute precision HH:MM")
    return parsed
