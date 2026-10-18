from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.core.timeframes import parse_timeframe_minutes
from backtest_engine.data.bar_availability import row_available_at
from backtest_engine.data.instrument_registry import InstrumentSpec
from backtest_engine.strategies.crossings import crossed as _crossed
from backtest_engine.strategies.resampling import IndicatorCache, ResampledHistory, aggregate_ohlcv
from backtest_engine.strategies.runtime import StrategyRuntimeContext


class IntradayStrategyContext:
    """Small helper API exposed to trusted intraday strategy plugins."""

    def __init__(
        self,
        history: pl.DataFrame,
        *,
        end_index: int | None = None,
        timeframe: str | None = None,
        bar_minutes: int | None = None,
        decision_time: time | None = None,
        previous_position: float = 0.0,
        order_statuses: list[dict[str, Any]] | None = None,
        indicator_cache: IndicatorCache | None = None,
        state: dict[str, Any] | None = None,
        equity_cash: float | None = None,
        point_value: float | None = None,
        position_quantity: float | None = None,
        instrument_spec: InstrumentSpec | None = None,
        base_currency: str = "BASE",
        available_funds: float | None = None,
    ) -> None:
        if history.height == 0:
            raise StrategyPluginError("Intraday strategy history must not be empty")
        if end_index is not None and (end_index < 0 or end_index >= history.height):
            msg = f"Intraday strategy end_index out of bounds: {end_index}"
            raise StrategyPluginError(msg)
        self._history = history
        self._end_index = history.height - 1 if end_index is None else end_index
        self.timeframe = timeframe or _single_timeframe(history)
        self.bar_minutes = (
            bar_minutes if bar_minutes is not None else _parse_timeframe(self.timeframe)
        )
        self.decision_time = decision_time
        self.previous_position = previous_position
        self._order_statuses = order_statuses or []
        self._indicator_cache = indicator_cache
        if indicator_cache is not None and indicator_cache.get(("source", "", 0)) is not history:
            indicator_cache.clear()
            indicator_cache[("source", "", 0)] = history
        self._current_bar = self._history.row(self._end_index, named=True)
        self.state = state if state is not None else {}
        self.equity_cash, self.point_value = equity_cash, point_value
        self.position_quantity = position_quantity
        self.instrument_spec = instrument_spec
        self.base_currency, self.available_funds = base_currency, available_funds

    def history(self) -> pl.DataFrame:
        return self._history.slice(0, self._visible_height())

    def history_tail(self, limit: int) -> pl.DataFrame:
        if limit <= 0:
            raise StrategyPluginError("history tail limit must be positive")
        height = min(limit, self._visible_height())
        return self._history.slice(self._visible_height() - height, height)

    @property
    def end_index(self) -> int:
        return self._end_index

    @property
    def history_source_token(self) -> int:
        """Opaque identity used by stateful runtimes to detect a new dataset."""
        return id(self._history)

    def session_history(self) -> pl.DataFrame:
        current_date = self.current_date
        timestamps = self._history.get_column("ts")
        start = self._end_index
        while start > 0 and _to_date(timestamps[start - 1]) == current_date:
            start -= 1
        return self._history.slice(start, self._end_index - start + 1)

    def current_bar(self) -> dict[str, Any]:
        return self._current_bar

    @property
    def current_ts(self) -> datetime:
        value = row_available_at(self._current_bar)
        if not isinstance(value, datetime):
            msg = f"ts must contain datetime values, got {value!r}"
            raise StrategyPluginError(msg)
        return value

    @property
    def current_time(self) -> time:
        return self.current_ts.time()

    @property
    def current_date(self) -> date:
        return self.current_ts.date()

    @property
    def active_contract(self) -> str | None:
        return _optional_current_string(self.current_bar(), "active_contract")

    @property
    def weekday(self) -> int:
        return self.current_ts.weekday()

    def series(self, column: str) -> pl.Series:
        if column not in self._history.columns:
            msg = f"Intraday strategy requested missing column: {column}"
            raise StrategyPluginError(msg)
        return self._history.get_column(column).slice(0, self._visible_height())

    def runtime(self, *, warmup_bars: int = 0) -> StrategyRuntimeContext:
        return StrategyRuntimeContext(
            self._history,
            current_index=self._end_index,
            warmup_bars=warmup_bars,
        )

    def order_statuses(self) -> list[dict[str, Any]]:
        return list(self._order_statuses)

    def sma(self, column: str, window: int) -> float | None:
        _validate_window(window, "sma")
        if self._visible_height() < window:
            return None
        values = self._history.get_column(column)
        dtype = values.dtype
        if not dtype.is_numeric():
            msg = f"sma column must be numeric, got {dtype}"
            raise StrategyPluginError(msg)
        cached = self._cached_indicator("sma", column, window)
        if cached is not None:
            value = cached[self._end_index]
            return None if value is None else float(value)
        tail = values.slice(self._visible_height() - window, window)
        if tail.null_count() > 0:
            return None
        return float(tail.mean())

    def ema(self, column: str, window: int) -> float | None:
        _validate_window(window, "ema")
        values = self.series(column)
        if len(values) < window:
            return None
        if values.tail(window).null_count() > 0:
            return None
        dtype = values.dtype
        if not dtype.is_numeric():
            msg = f"ema column must be numeric, got {dtype}"
            raise StrategyPluginError(msg)
        cached = self._cached_indicator("ema", column, window)
        if cached is not None:
            value = cached[self._end_index]
            return None if value is None else float(value)
        ema = values.ewm_mean(span=float(window), adjust=False, min_samples=window)
        value = ema[-1]
        return None if value is None else float(value)

    def bars_since_session_start(self) -> int:
        current_date = _to_date(self.current_bar()["ts"])
        count = 0
        ts = self._history.get_column("ts")
        for index in range(self._end_index, -1, -1):
            value = ts[index]
            if _to_date(value) != current_date:
                break
            count += 1
        return count

    def resample_ohlcv(self, every: str) -> pl.DataFrame:
        if self._indicator_cache is None:
            return aggregate_ohlcv(self.history(), every).drop_nulls("close")
        key = ("resample", every, 0)
        cached = self._indicator_cache.get(key)
        if not isinstance(cached, ResampledHistory):
            cached = ResampledHistory(self._history, every)
            self._indicator_cache[key] = cached
        return cached.update(self._end_index, every)

    def is_time(self, raw_time: str, tolerance_minutes: int | None = None) -> bool:
        target_time = _parse_time(raw_time)
        tolerance = self.bar_minutes if tolerance_minutes is None else tolerance_minutes
        if tolerance < 0:
            raise StrategyPluginError("tolerance_minutes must be non-negative")
        target_dt = datetime.combine(self.current_date, target_time)
        current_ts = self.current_ts
        if current_ts.tzinfo is not None:
            target_dt = target_dt.replace(tzinfo=current_ts.tzinfo)
        delta_minutes = (target_dt - current_ts).total_seconds() / 60.0
        return 0.0 <= delta_minutes <= tolerance

    def crossed_above(
        self,
        column: str,
        threshold_series_or_value: pl.Series | float | int,
        lookback_bars: int = 1,
    ) -> bool:
        return _crossed(self.series(column), threshold_series_or_value, lookback_bars, above=True)

    def crossed_below(
        self,
        column: str,
        threshold_series_or_value: pl.Series | float | int,
        lookback_bars: int = 1,
    ) -> bool:
        return _crossed(self.series(column), threshold_series_or_value, lookback_bars, above=False)

    def _visible_height(self) -> int:
        return self._end_index + 1

    def _cached_indicator(
        self,
        kind: str,
        column: str,
        window: int,
    ) -> pl.Series | None:
        if self._indicator_cache is None:
            return None
        key = (kind, column, window)
        cached = self._indicator_cache.get(key)
        if cached is None:
            values = self._history.get_column(column)
            if kind == "sma":
                cached = values.rolling_mean(window_size=window, min_samples=window)
            else:
                cached = values.ewm_mean(
                    span=float(window),
                    adjust=False,
                    min_samples=window,
                )
            self._indicator_cache[key] = cached
        return cached


def _validate_window(window: int, helper: str) -> None:
    if window <= 0:
        msg = f"{helper} window must be greater than 0"
        raise StrategyPluginError(msg)


def _single_timeframe(history: pl.DataFrame) -> str:
    if "timeframe" not in history.columns:
        raise StrategyPluginError("Intraday strategy history is missing timeframe")
    values = history.get_column("timeframe").drop_nulls().unique().to_list()
    if len(values) != 1:
        msg = "Intraday strategy history timeframe must contain exactly one non-null value"
        raise StrategyPluginError(msg)
    return str(values[0])


def _parse_timeframe(timeframe: str) -> int:
    try:
        return parse_timeframe_minutes(timeframe)
    except ValueError as exc:
        raise StrategyPluginError(str(exc)) from exc


def _parse_time(raw_time: str) -> time:
    try:
        parsed = time.fromisoformat(raw_time)
    except ValueError as exc:
        msg = f"Invalid time: {raw_time}. Expected HH:MM"
        raise StrategyPluginError(msg) from exc
    if parsed.second != 0 or parsed.microsecond != 0:
        msg = f"Invalid time: {raw_time}. Expected minute precision HH:MM"
        raise StrategyPluginError(msg)
    return parsed


def _optional_current_string(row: dict[str, Any], column: str) -> str | None:
    value = row.get(column)
    if value is None:
        return None
    return str(value)


def _to_date(value: Any) -> date:
    if hasattr(value, "date"):
        return value.date()
    msg = f"ts value must be datetime-like, got {value!r}"
    raise StrategyPluginError(msg)
