from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.intraday.event.reporting.config import (
    BEST_BY_COST_COLUMNS,
    BEST_BY_SCHEDULE_COLUMNS,
    TOP_RUN_COLUMNS,
)


def sort_frame(frame: pl.DataFrame, sort_by: str, descending: bool) -> pl.DataFrame:
    return frame.sort(sort_by, descending=descending, nulls_last=True)


def ranked_top_runs(summary: pl.DataFrame, top: int) -> pl.DataFrame:
    selected = summary.head(top).with_row_index("rank", offset=1)
    return select_existing_columns(selected, TOP_RUN_COLUMNS)


def best_by_schedule(
    summary: pl.DataFrame,
    sort_by: str,
    descending: bool,
) -> tuple[pl.DataFrame, list[str]]:
    required = {"schedule", "decision_time", sort_by}
    missing = sorted(required.difference(summary.columns))
    if missing or summary.height == 0:
        return pl.DataFrame(), [f"Skipped best_by_schedule: missing columns {', '.join(missing)}"]

    rows = []
    grouped = summary.group_by(["schedule", "decision_time"], maintain_order=True)
    for (_schedule, _decision_time), frame in grouped:
        rows.append(sort_frame(frame, sort_by, descending).row(0, named=True))
    return select_existing_columns(pl.DataFrame(rows, orient="row"), BEST_BY_SCHEDULE_COLUMNS), []


def best_by_cost(
    summary: pl.DataFrame,
    sort_by: str,
    descending: bool,
) -> tuple[pl.DataFrame, list[str]]:
    required = {"commission_bps", "slippage_bps", sort_by}
    missing = sorted(required.difference(summary.columns))
    if missing or summary.height == 0:
        return pl.DataFrame(), [f"Skipped best_by_cost: missing columns {', '.join(missing)}"]

    working = summary.with_columns(
        (pl.col("commission_bps").cast(pl.Float64) + pl.col("slippage_bps").cast(pl.Float64))
        .round(10)
        .alias("cost_level_bps")
    )
    rows = []
    for (cost_level,), frame in working.group_by("cost_level_bps", maintain_order=True):
        row = sort_frame(frame, sort_by, descending).row(0, named=True)
        row["cost_level_bps"] = cost_level
        rows.append(row)
    return select_existing_columns(pl.DataFrame(rows, orient="row"), BEST_BY_COST_COLUMNS), []


def risk_notes(summary: pl.DataFrame) -> str:
    if summary.height == 0 or "max_drawdown" not in summary.columns:
        return "_No drawdown data available._"
    worst = summary.sort("max_drawdown").head(min(5, summary.height))
    return markdown_table(
        select_existing_columns(
            worst,
            [
                "run_id",
                "decision_time",
                "window_minutes",
                "commission_bps",
                "slippage_bps",
                "total_return",
                "sharpe",
                "max_drawdown",
            ],
        )
    )


def exposure_notes(top_runs: pl.DataFrame) -> str:
    return markdown_table(
        select_existing_columns(
            top_runs,
            [
                "rank",
                "run_id",
                "decision_time",
                "window_minutes",
                "long_exposure",
                "short_exposure",
                "exposure",
            ],
        )
    )


def recommended_next_inspection(top_runs: pl.DataFrame) -> list[str]:
    if top_runs.height == 0 or "run_path" not in top_runs.columns:
        return ["- No successful run paths available."]
    row = top_runs.row(0, named=True)
    run_path = Path(str(row["run_path"]))
    return [
        f"- Best run events: `{run_path / 'events.csv'}` / `{run_path / 'events.parquet'}`",
        f"- Best run equity: `{run_path / 'equity.csv'}` / `{run_path / 'equity.parquet'}`",
        f"- Best run trades: `{run_path / 'trades.csv'}` / `{run_path / 'trades.parquet'}`",
        f"- Best run metrics: `{run_path / 'metrics.json'}`",
    ]


def markdown_table(frame: pl.DataFrame) -> str:
    if frame.height == 0 or not frame.columns:
        return "_No rows._"
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    lines = [header, divider]
    lines.extend(
        "| " + " | ".join(format_markdown_value(value) for value in row) + " |"
        for row in frame.iter_rows()
    )
    return "\n".join(lines)


def format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def experiment_id(
    summary: pl.DataFrame,
    experiment_config: dict[str, Any],
    experiment_path: Path,
) -> str:
    if "experiment_id" in summary.columns and summary.height > 0:
        return str(summary.get_column("experiment_id")[0])
    return str(experiment_config.get("experiment_id") or experiment_path.name)


def unique_strings(frame: pl.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(str(value) for value in frame.get_column(column).drop_nulls().unique())


def first_value(frame: pl.DataFrame, column: str) -> Any:
    if column not in frame.columns or frame.height == 0:
        return None
    return frame.get_column(column)[0]


def parameter_grid(experiment_config: dict[str, Any]) -> str:
    grid = experiment_config.get("grid")
    if not isinstance(grid, dict):
        return "n/a"
    return json.dumps(grid, sort_keys=True)


def select_existing_columns(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    existing = [column for column in columns if column in frame.columns]
    return frame.select(existing) if existing else pl.DataFrame()
