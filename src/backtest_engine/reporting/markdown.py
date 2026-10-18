from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

from backtest_engine.reporting.config import FAILED_COLUMNS, ReportConfig
from backtest_engine.reporting.tables import (
    experiment_id,
    first_value,
    low_trade_count_runs,
    markdown_table,
    parameter_grid,
    ranked_top_runs,
    recommended_next_inspection,
    select_existing_columns,
    sort_frame,
    unique_strings,
)


def render_markdown_report(
    *,
    config: ReportConfig,
    summary: pl.DataFrame,
    failed_runs: pl.DataFrame,
    top_runs: pl.DataFrame,
    best_by_dataset: pl.DataFrame,
    experiment_config: dict[str, Any],
    chart_records: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    lines = report_header(config, summary, failed_runs, experiment_config)
    lines.extend(top_run_section(top_runs))
    lines.extend(chart_section(chart_records, warnings))
    lines.extend(dataset_sections(config, summary))
    lines.extend(risk_sections(config, summary, best_by_dataset))
    lines.extend(failed_run_section(failed_runs))
    lines.extend(
        [
            "## Recommended Next Inspection",
            "",
            *recommended_next_inspection(best_by_dataset),
            "",
            (
                "Note: the report ranks historical runs. It does not perform walk-forward "
                "validation or out-of-sample testing."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def report_header(
    config: ReportConfig,
    summary: pl.DataFrame,
    failed_runs: pl.DataFrame,
    experiment_config: dict[str, Any],
) -> list[str]:
    datasets = unique_strings(summary, "continuous_symbol")
    strategy_file = first_value(summary, "strategy_file") or experiment_config.get(
        "strategy_file", ""
    )
    return [
        "# Backtest Experiment Report",
        "",
        f"- experiment_id: `{experiment_id(summary, experiment_config, config.experiment_path)}`",
        f"- generated_at: `{datetime.now(UTC).isoformat()}`",
        f"- experiment_path: `{config.experiment_path}`",
        "",
        "## Overview",
        "",
        f"- successful_runs: {summary.height}",
        f"- failed_runs: {failed_runs.height}",
        f"- datasets: {', '.join(datasets) if datasets else 'n/a'}",
        f"- strategy_file: `{strategy_file}`",
        f"- parameter_grid: `{parameter_grid(experiment_config)}`",
        f"- commission_bps: {first_value(summary, 'commission_bps')}",
        f"- slippage_bps: {first_value(summary, 'slippage_bps')}",
        "",
    ]


def top_run_section(top_runs: pl.DataFrame) -> list[str]:
    return ["## Top Runs Overall", "", markdown_table(top_runs), ""]


def chart_section(chart_records: list[dict[str, Any]], warnings: list[str]) -> list[str]:
    lines = ["## Charts", ""]
    if chart_records:
        for chart in chart_records:
            title = str(chart.get("title") or chart.get("chart_type") or "Chart")
            path = str(chart["path"]).replace("\\", "/")
            lines.extend([f"[![{title}]({path})]({path})", ""])
    else:
        lines.extend(["No charts were generated.", ""])
    if warnings:
        lines.extend(["### Chart Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    return lines


def dataset_sections(config: ReportConfig, summary: pl.DataFrame) -> list[str]:
    if not config.group_by_dataset or "continuous_symbol" not in summary.columns:
        return []
    lines = ["## Top Runs By Dataset", ""]
    for dataset in unique_strings(summary, "continuous_symbol"):
        dataset_summary = summary.filter(pl.col("continuous_symbol") == dataset)
        lines.extend(
            [
                f"### {dataset}",
                "",
                markdown_table(
                    ranked_top_runs(
                        sort_frame(dataset_summary, config.sort_by, config.descending),
                        config.top,
                    )
                ),
                "",
            ]
        )
    return lines


def risk_sections(
    config: ReportConfig,
    summary: pl.DataFrame,
    best_by_dataset: pl.DataFrame,
) -> list[str]:
    return [
        "## Best Run Per Dataset",
        "",
        markdown_table(best_by_dataset),
        "",
        "## Risk Notes",
        "",
        "### Worst Max Drawdown Runs",
        "",
        markdown_table(worst_drawdown_runs(summary, config.top)),
        "",
        "### Negative Total Return Runs",
        "",
        markdown_table(negative_return_runs(summary)),
        "",
        "### Low Trade Count Runs",
        "",
        markdown_table(low_trade_count_runs(summary)),
        "",
        "## Failed Runs",
        "",
    ]


def failed_run_section(failed_runs: pl.DataFrame) -> list[str]:
    if failed_runs.height == 0:
        return ["No failed runs.", ""]
    return [markdown_table(select_existing_columns(failed_runs, FAILED_COLUMNS)), ""]


def worst_drawdown_runs(summary: pl.DataFrame, top: int) -> pl.DataFrame:
    if "max_drawdown" not in summary.columns:
        return pl.DataFrame()
    return select_existing_columns(
        summary.sort("max_drawdown").head(min(top, summary.height)),
        [
            "continuous_symbol",
            "run_id",
            "fast_window",
            "slow_window",
            "max_drawdown",
            "total_return",
            "sharpe",
        ],
    )


def negative_return_runs(summary: pl.DataFrame) -> pl.DataFrame:
    if "total_return" not in summary.columns:
        return pl.DataFrame()
    return select_existing_columns(
        summary.filter(pl.col("total_return") < 0),
        [
            "continuous_symbol",
            "run_id",
            "fast_window",
            "slow_window",
            "total_return",
            "sharpe",
        ],
    )
