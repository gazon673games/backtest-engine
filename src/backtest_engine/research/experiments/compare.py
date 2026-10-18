from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from backtest_engine.research.experiments.config import (
    BEST_BY_GROUP_COLUMNS,
    TOP_RUN_COLUMNS,
    ExperimentCompareConfig,
)
from backtest_engine.research.experiments.utils import (
    markdown_table,
    select_existing_columns,
    status_count,
    unique_strings,
)


def sort_frame(frame: pl.DataFrame, sort_by: str, descending: bool) -> pl.DataFrame:
    return frame.sort(sort_by, descending=descending, nulls_last=True)


def ranked_top_runs(summary: pl.DataFrame, top: int) -> pl.DataFrame:
    selected = summary.head(top).with_row_index("rank", offset=1)
    return select_existing_columns(selected, TOP_RUN_COLUMNS)


def best_by_group(
    registry: pl.DataFrame,
    group_by: list[str],
    sort_by: str,
    descending: bool,
) -> tuple[pl.DataFrame, list[str]]:
    missing = sorted(column for column in group_by if column not in registry.columns)
    if missing:
        return pl.DataFrame(), [f"Skipped best_by_group: missing columns {', '.join(missing)}"]

    rows = []
    for key_values, frame in registry.group_by(group_by, maintain_order=True):
        key_tuple = key_values if isinstance(key_values, tuple) else (key_values,)
        row = sort_frame(frame, sort_by, descending).row(0, named=True)
        row["group_key"] = "|".join(
            f"{column}={value}" for column, value in zip(group_by, key_tuple, strict=True)
        )
        rows.append(row)
    return select_existing_columns(pl.DataFrame(rows, orient="row"), BEST_BY_GROUP_COLUMNS), []


def render_compare_report(
    *,
    config: ExperimentCompareConfig,
    registry: pl.DataFrame,
    top_runs: pl.DataFrame,
    best_by_group: pl.DataFrame,
    warnings: list[str],
) -> str:
    group_by = config.group_by or ["continuous_symbol", "timeframe", "strategy_sha256"]
    lines = [
        "# Experiment Compare Report",
        "",
        f"- generated_at: `{datetime.now(UTC).isoformat()}`",
        f"- registry_path: `{config.registry_path}`",
        f"- root: `{config.root}`",
        f"- sort_by: `{config.sort_by}`",
        f"- descending: `{config.descending}`",
        f"- group_by: `{', '.join(group_by)}`",
        "",
        "## Overview",
        "",
        *overview_lines(registry),
        "",
        "## Top Runs",
        "",
        markdown_table(top_runs),
        "",
        "## Best Run By Group",
        "",
        markdown_table(best_by_group),
        "",
        "## Cost / Friction Notes",
        "",
        markdown_table(cost_friction_table(top_runs)),
        "",
        "## Failed Runs",
        "",
        markdown_table(failed_runs_table(registry)),
        "",
    ]
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    return "\n".join(lines)


def overview_lines(registry: pl.DataFrame) -> list[str]:
    datasets = unique_strings(registry, "continuous_symbol")
    strategies = unique_strings(registry, "strategy_sha256")
    return [
        f"- total_rows: {registry.height}",
        f"- successful_runs: {status_count(registry, 'success')}",
        f"- failed_runs: {status_count(registry, 'failed')}",
        f"- datasets: {', '.join(datasets) if datasets else 'n/a'}",
        f"- strategy_hashes: {len(strategies)}",
    ]


def cost_friction_table(top_runs: pl.DataFrame) -> pl.DataFrame:
    return select_existing_columns(
        top_runs,
        [
            "rank",
            "run_id",
            "commission_bps",
            "slippage_bps",
            "total_cost_cash",
            "turnover_notional",
            "total_return",
            "sharpe",
        ],
    )


def failed_runs_table(registry: pl.DataFrame) -> pl.DataFrame:
    return select_existing_columns(
        registry.filter(pl.col("status") == "failed"),
        [
            "parent_experiment_id",
            "dataset_path",
            "strategy_file",
            "error_type",
            "error_message",
        ],
    )
