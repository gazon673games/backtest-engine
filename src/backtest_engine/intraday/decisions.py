from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.timeframes import parse_timeframe_minutes
from backtest_engine.intraday.config import IntradayBacktestConfig, IntradayExecutionConfig
from backtest_engine.intraday.utils import as_datetime
from backtest_engine.intraday.validation import single_value
from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.plugins.loader import (
    generate_intraday_plugin_signal_from_callable,
    load_intraday_strategy_plugin,
)


@dataclass(frozen=True)
class IndexedRow:
    index: int
    row: dict[str, Any]


def build_decision_points(
    df: pl.DataFrame,
    config: IntradayBacktestConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if config.execution.decision_time is None:
        raise DatasetValidationError("execution.decision_time is required for a decision-point run")
    timeframe = single_value(df, "timeframe")
    try:
        bar_minutes = parse_timeframe_minutes(timeframe)
    except ValueError as exc:
        raise DatasetValidationError(str(exc)) from exc
    tolerance_minutes = effective_decision_tolerance_minutes(config.execution, bar_minutes)
    rows_by_date = rows_by_trading_date(df)
    decision_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    ts_dtype = df.schema["ts"]
    generate_signal = load_intraday_strategy_plugin(
        config.strategy.strategy_file, params=config.strategy.strategy_params
    )

    for trading_date in sorted(rows_by_date):
        if trading_date < config.execution.from_date:
            continue
        selected = select_decision_row(
            trading_date,
            rows_by_date[trading_date],
            config.execution.decision_time,
            tolerance_minutes,
        )
        if selected is None:
            _skip_decision_day(
                trading_date,
                rows_by_date[trading_date],
                config.execution,
                tolerance_minutes,
                decision_rows,
                skipped_rows,
            )
            continue

        ctx = IntradayStrategyContext(
            df,
            end_index=selected.index,
            timeframe=timeframe,
            bar_minutes=bar_minutes,
            decision_time=config.execution.decision_time,
        )
        bars_since_session_start = ctx.bars_since_session_start()
        if bars_since_session_start < config.execution.minimum_bars_since_session_start:
            _skip_decision_day(
                trading_date,
                rows_by_date[trading_date],
                config.execution,
                tolerance_minutes,
                decision_rows,
                skipped_rows,
                reason="insufficient_bars_since_session_start",
                selected_decision=selected,
                selected_bars_since_session_start=bars_since_session_start,
            )
            continue

        plugin_signal = generate_intraday_plugin_signal_from_callable(
            generate_signal,
            ctx,
            config.strategy.strategy_params,
        )
        decision_rows.append(
            _decision_row(selected.row, trading_date, plugin_signal, bars_since_session_start)
        )

    return (
        pl.DataFrame(decision_rows, schema=decision_points_schema(ts_dtype), orient="row"),
        pl.DataFrame(skipped_rows, schema=skipped_days_schema(ts_dtype), orient="row"),
    )


def resolve_intraday_config(
    config: IntradayBacktestConfig,
    dataset_validation: dict[str, Any],
) -> IntradayBacktestConfig:
    bar_minutes = int(dataset_validation["bar_minutes"])
    execution = config.execution.model_copy(
        update={
            "decision_tolerance_minutes": effective_decision_tolerance_minutes(
                config.execution,
                bar_minutes,
            )
        }
    )
    return config.model_copy(update={"execution": execution})


def effective_decision_tolerance_minutes(
    execution: IntradayExecutionConfig,
    bar_minutes: int,
) -> int:
    if execution.decision_tolerance_minutes is None:
        return bar_minutes
    return execution.decision_tolerance_minutes


def rows_by_trading_date(df: pl.DataFrame) -> dict[date, list[IndexedRow]]:
    rows_by_date: dict[date, list[IndexedRow]] = {}
    for index, row in enumerate(df.iter_rows(named=True)):
        ts = as_datetime(row["ts"])
        rows_by_date.setdefault(ts.date(), []).append(IndexedRow(index=index, row=row))
    return rows_by_date


def select_decision_row(
    trading_date: date,
    rows: list[IndexedRow],
    decision_time: time,
    tolerance_minutes: int,
) -> IndexedRow | None:
    if not rows:
        return None
    sample_ts = as_datetime(rows[0].row["ts"])
    decision_dt = datetime.combine(trading_date, decision_time)
    if sample_ts.tzinfo is not None:
        decision_dt = decision_dt.replace(tzinfo=sample_ts.tzinfo)
    lower_bound = decision_dt - timedelta(minutes=tolerance_minutes)
    candidates = [
        indexed_row
        for indexed_row in rows
        if lower_bound <= as_datetime(indexed_row.row["ts"]) <= decision_dt
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda indexed_row: as_datetime(indexed_row.row["ts"]))


def skipped_decision_row(
    trading_date: date,
    rows: list[IndexedRow],
    decision_time: time,
    tolerance_minutes: int,
    minimum_bars_since_session_start: int,
    previous_decision_ts: Any | None,
    *,
    reason: str = "missing_decision_bar_within_tolerance",
    selected_decision: IndexedRow | None = None,
    selected_bars_since_session_start: int | None = None,
) -> dict[str, Any]:
    sample_ts = as_datetime(rows[0].row["ts"])
    decision_dt = datetime.combine(trading_date, decision_time)
    if sample_ts.tzinfo is not None:
        decision_dt = decision_dt.replace(tzinfo=sample_ts.tzinfo)
    before = [row for row in rows if as_datetime(row.row["ts"]) <= decision_dt]
    after = [row for row in rows if as_datetime(row.row["ts"]) > decision_dt]
    return {
        "trading_date": trading_date,
        "reason": reason,
        "decision_time": decision_time.isoformat(timespec="minutes"),
        "decision_tolerance_minutes": tolerance_minutes,
        "minimum_bars_since_session_start": minimum_bars_since_session_start,
        "previous_decision_ts": previous_decision_ts,
        "selected_decision_ts": (
            selected_decision.row["ts"] if selected_decision is not None else None
        ),
        "selected_bars_since_session_start": selected_bars_since_session_start,
        "nearest_before_ts": _nearest_before(before),
        "nearest_after_ts": _nearest_after(after),
    }


def decision_points_schema(ts_dtype: pl.DataType) -> dict[str, pl.DataType]:
    return {
        "trading_date": pl.Date,
        "decision_ts": ts_dtype,
        "decision_close": pl.Float64,
        "plugin_signal": pl.Float64,
        "target_position": pl.Float64,
        "active_contract": pl.String,
        "source_contract_symbol": pl.String,
        "bars_since_session_start": pl.Int64,
        "forced_flat": pl.Boolean,
        "forced_flat_reason": pl.String,
    }


def skipped_days_schema(ts_dtype: pl.DataType) -> dict[str, pl.DataType]:
    return {
        "trading_date": pl.Date,
        "reason": pl.String,
        "decision_time": pl.String,
        "decision_tolerance_minutes": pl.Int64,
        "minimum_bars_since_session_start": pl.Int64,
        "previous_decision_ts": ts_dtype,
        "selected_decision_ts": ts_dtype,
        "selected_bars_since_session_start": pl.Int64,
        "nearest_before_ts": ts_dtype,
        "nearest_after_ts": ts_dtype,
    }


def _skip_decision_day(
    trading_date: date,
    rows: list[IndexedRow],
    execution: IntradayExecutionConfig,
    tolerance_minutes: int,
    decision_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    **row_overrides: Any,
) -> None:
    skipped_rows.append(
        skipped_decision_row(
            trading_date,
            rows,
            execution.decision_time,
            tolerance_minutes,
            execution.minimum_bars_since_session_start,
            decision_rows[-1]["decision_ts"] if decision_rows else None,
            **row_overrides,
        )
    )


def _decision_row(
    row: dict[str, Any],
    trading_date: date,
    plugin_signal: float,
    bars_since_session_start: int,
) -> dict[str, Any]:
    return {
        "trading_date": trading_date,
        "decision_ts": row["ts"],
        "decision_close": float(row["close"]),
        "plugin_signal": plugin_signal,
        "target_position": plugin_signal,
        "active_contract": str(row["active_contract"]),
        "source_contract_symbol": str(row["source_contract_symbol"]),
        "bars_since_session_start": bars_since_session_start,
        "forced_flat": False,
        "forced_flat_reason": None,
    }


def _nearest_before(rows: list[IndexedRow]) -> Any | None:
    if not rows:
        return None
    return max(rows, key=lambda indexed_row: as_datetime(indexed_row.row["ts"])).row["ts"]


def _nearest_after(rows: list[IndexedRow]) -> Any | None:
    if not rows:
        return None
    return min(rows, key=lambda indexed_row: as_datetime(indexed_row.row["ts"])).row["ts"]
