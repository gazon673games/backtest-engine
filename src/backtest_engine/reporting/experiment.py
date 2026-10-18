"""Builds ranked experiment reports from reproducible run artifacts."""

from __future__ import annotations

from typing import Any

import polars as pl

from backtest_engine.core.exceptions import ReportingError
from backtest_engine.reporting.config import OutputFormat, ReportConfig, ReportResult
from backtest_engine.reporting.experiment_charts import generate_charts
from backtest_engine.reporting.io import (
    output_formats,
    read_optional_json,
    read_optional_table,
    read_required_table,
    write_named_outputs,
)
from backtest_engine.reporting.markdown import render_markdown_report
from backtest_engine.reporting.tables import best_by_dataset, ranked_top_runs, sort_frame

__all__ = [
    "OutputFormat",
    "ReportConfig",
    "ReportResult",
    "generate_experiment_report",
]


def generate_experiment_report(config: ReportConfig) -> ReportResult:
    experiment_path = config.experiment_path
    summary = read_required_table(experiment_path, "summary")
    failed_runs = read_optional_table(experiment_path, "failed_runs")
    experiment_config = read_optional_json(experiment_path / "experiment_config.json")

    if config.sort_by not in summary.columns:
        msg = f"sort_by column does not exist in summary: {config.sort_by}"
        raise ReportingError(msg)

    sorted_summary = sort_frame(summary, config.sort_by, config.descending)
    top_runs = ranked_top_runs(sorted_summary, config.top)
    dataset_best = best_by_dataset(summary, config.sort_by, config.descending)
    chart_records, warnings = maybe_generate_charts(config, summary, top_runs, dataset_best)
    formats = output_formats(config.output_format)

    report_path = maybe_write_markdown(
        config=config,
        summary=summary,
        failed_runs=failed_runs,
        top_runs=top_runs,
        best_by_dataset=dataset_best,
        experiment_config=experiment_config,
        chart_records=chart_records,
        warnings=warnings,
        formats=formats,
    )
    top_runs_path = write_table_outputs(experiment_path / "top_runs", top_runs, formats)
    best_by_dataset_path = write_table_outputs(
        experiment_path / "best_by_dataset",
        dataset_best,
        formats,
    )
    return ReportResult(
        experiment_path=experiment_path,
        report_path=report_path,
        top_runs_path=top_runs_path,
        best_by_dataset_path=best_by_dataset_path,
        top_run_count=top_runs.height,
        failed_run_count=failed_runs.height,
        best_overall=top_runs.row(0, named=True) if top_runs.height > 0 else None,
        best_by_dataset=[row for row in dataset_best.iter_rows(named=True)],
        chart_paths=[str(record["path"]) for record in chart_records],
        warnings=warnings,
    )


def maybe_generate_charts(
    config: ReportConfig,
    summary: pl.DataFrame,
    top_runs: pl.DataFrame,
    dataset_best: pl.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not config.charts:
        return [], []
    return generate_charts(
        config=config,
        summary=summary,
        top_runs=top_runs,
        best_by_dataset=dataset_best,
    )


def maybe_write_markdown(
    *,
    config: ReportConfig,
    summary: pl.DataFrame,
    failed_runs: pl.DataFrame,
    top_runs: pl.DataFrame,
    best_by_dataset: pl.DataFrame,
    experiment_config: dict[str, Any],
    chart_records: list[dict[str, Any]],
    warnings: list[str],
    formats: set[str],
) -> Any:
    if "md" not in formats:
        return None
    report_path = config.experiment_path / "report.md"
    report_path.write_text(
        render_markdown_report(
            config=config,
            summary=summary,
            failed_runs=failed_runs,
            top_runs=top_runs,
            best_by_dataset=best_by_dataset,
            experiment_config=experiment_config,
            chart_records=chart_records,
            warnings=warnings,
        ),
        encoding="utf-8",
    )
    return report_path


def write_table_outputs(base_path: Any, frame: pl.DataFrame, formats: set[str]) -> Any:
    write_named_outputs(base_path, frame, formats)
    for output_format in ("csv", "parquet", "json"):
        if output_format in formats:
            return base_path.with_suffix(f".{output_format}")
    return None
