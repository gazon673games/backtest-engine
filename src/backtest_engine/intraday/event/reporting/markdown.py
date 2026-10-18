from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

from backtest_engine.intraday.event.reporting.config import (
    FAILED_COLUMNS,
    IntradayEventReportConfig,
)
from backtest_engine.intraday.event.reporting.tables import (
    experiment_id,
    exposure_notes,
    first_value,
    markdown_table,
    parameter_grid,
    recommended_next_inspection,
    risk_notes,
    select_existing_columns,
    unique_strings,
)


def render_markdown_report(
    *,
    config: IntradayEventReportConfig,
    summary: pl.DataFrame,
    failed_runs: pl.DataFrame,
    top_runs: pl.DataFrame,
    best_by_schedule: pl.DataFrame,
    best_by_cost: pl.DataFrame,
    experiment_config: dict[str, Any],
    chart_records: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    lines = header(config, summary, failed_runs, experiment_config)
    lines.extend(grid_sections(top_runs, best_by_schedule, best_by_cost, experiment_config))
    lines.extend(chart_section(chart_records, warnings))
    lines.extend(risk_sections(summary, top_runs))
    lines.extend(failed_runs_section(failed_runs))
    lines.extend(["## Recommended Next Inspection", "", *recommended_next_inspection(top_runs), ""])
    return "\n".join(lines)


def header(
    config: IntradayEventReportConfig,
    summary: pl.DataFrame,
    failed_runs: pl.DataFrame,
    experiment_config: dict[str, Any],
) -> list[str]:
    datasets = unique_strings(summary, "continuous_symbol")
    strategy_file = first_value(summary, "strategy_file") or experiment_config.get(
        "strategy_file", ""
    )
    return [
        "# Intraday Event Experiment Report",
        "",
        f"- experiment_id: `{experiment_id(summary, experiment_config, config.experiment_path)}`",
        f"- generated_at: `{datetime.now(UTC).isoformat()}`",
        f"- experiment_path: `{config.experiment_path}`",
        "",
        "## Overview",
        "",
        f"- successful_runs: {summary.height}",
        f"- failed_runs: {failed_runs.height}",
        f"- sort_by: `{config.sort_by}`",
        f"- top: {config.top}",
        "",
        "Top in-sample Sharpe from a parameter grid is not proof of robustness.",
        "This report does not perform walk-forward or out-of-sample validation.",
        "",
        "## Dataset",
        "",
        f"- datasets: {', '.join(datasets) if datasets else 'n/a'}",
        f"- timeframe: {first_value(summary, 'timeframe')}",
        f"- dataset_type: {first_value(summary, 'dataset_type')}",
        f"- adjustment_mode: {first_value(summary, 'adjustment_mode')}",
        f"- from_date: {first_value(summary, 'from_date')}",
        "",
        "## Strategy",
        "",
        f"- strategy_file: `{strategy_file}`",
        f"- strategy_file_sha256: `{first_value(summary, 'strategy_file_sha256')}`",
        "",
    ]


def grid_sections(
    top_runs: pl.DataFrame,
    best_by_schedule: pl.DataFrame,
    best_by_cost: pl.DataFrame,
    experiment_config: dict[str, Any],
) -> list[str]:
    return [
        "## Grid",
        "",
        f"`{parameter_grid(experiment_config)}`",
        "",
        "## Top Runs Overall",
        "",
        markdown_table(top_runs),
        "",
        "## Best Runs by Decision Time",
        "",
        markdown_table(best_by_schedule),
        "",
        "## Best Runs by Cost Level",
        "",
        markdown_table(best_by_cost),
        "",
    ]


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


def risk_sections(summary: pl.DataFrame, top_runs: pl.DataFrame) -> list[str]:
    return [
        "## Risk / Drawdown Notes",
        "",
        risk_notes(summary),
        "",
        "## Long vs Short Exposure Notes",
        "",
        exposure_notes(top_runs),
        "",
        "## Failed Runs",
        "",
    ]


def failed_runs_section(failed_runs: pl.DataFrame) -> list[str]:
    if failed_runs.height == 0:
        return ["No failed runs.", ""]
    return [markdown_table(select_existing_columns(failed_runs, FAILED_COLUMNS)), ""]
