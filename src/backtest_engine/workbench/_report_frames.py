from __future__ import annotations

import math
from typing import Any

import polars as pl

from backtest_engine.workbench._report_values import (
    json_value,
    parse_features,
)


def cost_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return selected_frame(
        frame,
        (
            "cost_cash",
            "commission_cost_cash",
            "slippage_cost_cash",
            "spread_cost_cash",
            "roll_cost_cash",
        ),
        timestamp_column="fill_ts",
    )


def selected_frame(
    frame: pl.DataFrame,
    wanted: tuple[str, ...],
    *,
    timestamp_column: str = "ts",
) -> pl.DataFrame:
    if frame.height == 0:
        return frame
    time_column = timestamp_column if timestamp_column in frame.columns else "ts"
    columns = [time_column, *(name for name in wanted if name in frame.columns)]
    result = frame.select(columns)
    return result.rename({time_column: "ts"}) if time_column != "ts" else result


def frame_records(
    frame: pl.DataFrame,
    max_points: int,
    *,
    extrema_column: str | None = None,
    transition_columns: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if frame.height == 0:
        return []
    if transition_columns:
        transitions = transition_frame(frame, transition_columns)
        sampled = downsample(transitions, max_points)
    elif extrema_column and extrema_column in frame.columns:
        sampled = extrema_downsample(frame, max_points, extrema_column)
    else:
        sampled = downsample(frame, max_points)
    return table_records(sampled)


def table_records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return [{key: json_value(value) for key, value in row.items()} for row in frame.to_dicts()]


def downsample(frame: pl.DataFrame, max_points: int) -> pl.DataFrame:
    limit = max(1, max_points)
    if frame.height <= limit:
        return frame
    if limit == 1:
        return frame.tail(1)
    step = max(1, math.ceil((frame.height - 1) / (limit - 1)))
    indices = list(range(0, frame.height, step))
    if indices[-1] != frame.height - 1:
        indices.append(frame.height - 1)
    return select_rows(frame, indices)


def transition_frame(frame: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    present = [column for column in columns if column in frame.columns]
    if frame.height <= 1 or not present:
        return frame
    states = frame.select(present).rows()
    indices = [0]
    indices.extend(index for index in range(1, frame.height) if states[index] != states[index - 1])
    if indices[-1] != frame.height - 1:
        indices.append(frame.height - 1)
    return select_rows(frame, indices)


def extrema_downsample(frame: pl.DataFrame, max_points: int, column: str) -> pl.DataFrame:
    limit = max(1, max_points)
    if frame.height <= limit or limit < 4:
        return downsample(frame, limit)
    bucket_count = max(1, (limit - 2) // 2)
    bucket_size = max(1, math.ceil((frame.height - 2) / bucket_count))
    values = frame.get_column(column).to_list()
    indices = {0, frame.height - 1}
    for start in range(1, frame.height - 1, bucket_size):
        end = min(frame.height - 1, start + bucket_size)
        finite_values = [
            (index, float(values[index]))
            for index in range(start, end)
            if values[index] is not None and math.isfinite(float(values[index]))
        ]
        if finite_values:
            indices.add(min(finite_values, key=lambda item: item[1])[0])
            indices.add(max(finite_values, key=lambda item: item[1])[0])
    return select_rows(frame, sorted(indices))


def select_rows(frame: pl.DataFrame, indices: list[int]) -> pl.DataFrame:
    return frame.with_row_index("_row").filter(pl.col("_row").is_in(indices)).drop("_row")


def extract_plot_series(
    events: pl.DataFrame,
    *,
    max_points: int = 5000,
) -> dict[str, list[dict[str, Any]]]:
    if events.height == 0 or "strategy_features" not in events.columns:
        return {}
    plots: dict[str, list[dict[str, Any]]] = {}
    candidates = events.select("ts", "strategy_features").filter(
        pl.col("strategy_features").is_not_null()
        & (pl.col("strategy_features") != "")
        & (pl.col("strategy_features") != "{}")
    )
    for row in candidates.iter_rows(named=True):
        features = parse_features(row.get("strategy_features"))
        for plot in features.get("plots", []) if isinstance(features, dict) else []:
            if not isinstance(plot, dict) or "name" not in plot:
                continue
            name = str(plot["name"])
            plots.setdefault(name, []).append(
                {
                    "ts": json_value(row.get("ts")),
                    "type": str(plot.get("type", "line")),
                    "value": json_value(plot.get("value")),
                }
            )
    return {name: downsample_records(records, max_points) for name, records in plots.items()}


def downsample_records(
    records: list[dict[str, Any]],
    max_points: int,
) -> list[dict[str, Any]]:
    limit = max(1, max_points)
    if len(records) <= limit:
        return records
    if limit == 1:
        return records[-1:]
    step = max(1, math.ceil((len(records) - 1) / (limit - 1)))
    sampled = records[::step]
    if sampled[-1] is not records[-1]:
        sampled.append(records[-1])
    return sampled
