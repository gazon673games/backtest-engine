from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.reporting.config import BEST_RUN_COLUMNS, TOP_RUN_COLUMNS


def sort_frame(frame: pl.DataFrame, sort_by: str, descending: bool) -> pl.DataFrame:
    return frame.sort(sort_by, descending=descending, nulls_last=True)


def ranked_top_runs(summary: pl.DataFrame, top: int) -> pl.DataFrame:
    selected = summary.head(top).with_row_index("rank", offset=1)
    return select_existing_columns(selected, TOP_RUN_COLUMNS)


def best_by_dataset(summary: pl.DataFrame, sort_by: str, descending: bool) -> pl.DataFrame:
    if "continuous_symbol" not in summary.columns or summary.height == 0:
        return pl.DataFrame()

    rows = []
    for dataset in sorted(str(value) for value in summary.get_column("continuous_symbol").unique()):
        dataset_frame = summary.filter(pl.col("continuous_symbol") == dataset)
        rows.append(sort_frame(dataset_frame, sort_by, descending).row(0, named=True))
    return select_existing_columns(pl.DataFrame(rows, orient="row"), BEST_RUN_COLUMNS)


def select_existing_columns(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    existing = [column for column in columns if column in frame.columns]
    return frame.select(existing) if existing else pl.DataFrame()


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


def low_trade_count_runs(summary: pl.DataFrame) -> pl.DataFrame:
    if "trade_count" not in summary.columns or summary.height == 0:
        return pl.DataFrame()

    threshold = max(1, int(summary.get_column("trade_count").median() or 0))
    return select_existing_columns(
        summary.filter(pl.col("trade_count") <= threshold).sort("trade_count"),
        [
            "continuous_symbol",
            "run_id",
            "fast_window",
            "slow_window",
            "trade_count",
            "total_return",
            "sharpe",
        ],
    )


def recommended_next_inspection(best_runs: pl.DataFrame) -> list[str]:
    if best_runs.height == 0 or "run_path" not in best_runs.columns:
        return ["- No successful run paths available."]

    lines: list[str] = []
    for row in best_runs.iter_rows(named=True):
        symbol = row.get("continuous_symbol", "dataset")
        run_path = Path(str(row["run_path"]))
        lines.extend(
            [
                f"- {symbol} equity: `{run_path / 'equity.csv'}` / `{run_path / 'equity.parquet'}`",
                f"- {symbol} trades: `{run_path / 'trades.csv'}` / `{run_path / 'trades.parquet'}`",
                f"- {symbol} metrics: `{run_path / 'metrics.json'}`",
            ]
        )
    return lines
