"""Synchronized event timeline construction for portfolio legs."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.bar_availability import row_available_at
from backtest_engine.execution.model import event_loop_stop, execution_interval
from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig


def timestamps_by_asset(
    frames: dict[str, pl.DataFrame],
) -> dict[str, list[datetime]]:
    timestamps = {}
    for asset_id, frame in frames.items():
        columns = [
            column for column in ("available_at", "bar_close_ts", "ts") if column in frame.columns
        ]
        timestamps[asset_id] = [
            row_available_at(row) for row in frame.select(columns).iter_rows(named=True)
        ]
    return timestamps


def exact_event_indices(
    timestamps: dict[str, list[datetime]],
    timing: str,
) -> dict[str, dict[datetime, int]]:
    return {
        asset_id: {
            ts: index for index, ts in enumerate(asset_ts[: event_loop_stop(len(asset_ts), timing)])
        }
        for asset_id, asset_ts in timestamps.items()
    }


def event_times(
    timestamps: dict[str, list[datetime]],
    config: MultiInstrumentBacktestConfig,
) -> list[datetime]:
    return sorted(
        {
            ts
            for asset_ts in timestamps.values()
            for ts in asset_ts[:-1]
            if ts.date() >= config.execution.from_date
        }
    )


def validate_execution_timeline(
    exact_indices: dict[str, dict[datetime, int]],
    rows_by_asset: dict[str, Sequence[dict[str, Any]]],
    timing: str,
) -> None:
    """Fail closed when eager interval settlement would expose future state."""
    decision_times = sorted(
        {timestamp for indices in exact_indices.values() for timestamp in indices}
    )
    previous_exit: datetime | None = None
    for current_ts in decision_times:
        intervals = [
            execution_interval(rows_by_asset[asset_id], indices[current_ts], timing)
            for asset_id, indices in exact_indices.items()
            if current_ts in indices
        ]
        exit_times = {interval.exit_ts for interval in intervals}
        if len(exit_times) > 1:
            raise DatasetValidationError(
                "Multi-instrument execution requires synchronized interval exit timestamps; "
                f"decision_ts={current_ts.isoformat()}"
            )
        exit_ts = next(iter(exit_times))
        if previous_exit is not None and exit_ts < previous_exit:
            raise DatasetValidationError(
                "Multi-instrument execution intervals settle out of chronological order; "
                "align leg timestamps or use a common prepared timeframe"
            )
        previous_exit = exit_ts
        next_index = bisect_right(decision_times, current_ts)
        next_decision = decision_times[next_index] if next_index < len(decision_times) else None
        if next_decision is None:
            continue
        for interval in intervals:
            if current_ts < next_decision < interval.fill_ts:
                raise DatasetValidationError(
                    "Multi-instrument execution would expose a future position before fill_ts; "
                    "align leg timestamps or use signal_close_to_same_close"
                )
