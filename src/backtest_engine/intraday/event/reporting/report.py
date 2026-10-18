from __future__ import annotations

from typing import Any

import polars as pl

from backtest_engine.core.exceptions import ReportingError
from backtest_engine.intraday.event.reporting.charts import generate_charts
from backtest_engine.intraday.event.reporting.config import (
    IntradayEventReportConfig,
    IntradayEventReportResult,
)
from backtest_engine.intraday.event.reporting.io import (
    read_optional_json,
    read_optional_table,
    read_required_table,
    write_named_outputs,
)
from backtest_engine.intraday.event.reporting.markdown import render_markdown_report
from backtest_engine.intraday.event.reporting.tables import (
    best_by_cost,
    best_by_schedule,
    ranked_top_runs,
    sort_frame,
)

__all__ = [
    "IntradayEventReportConfig",
    "IntradayEventReportResult",
    "generate_intraday_event_report",
]


def generate_intraday_event_report(
    config: IntradayEventReportConfig,
) -> IntradayEventReportResult:
    summary = read_required_table(config.experiment_path, "summary")
    failed_runs = read_optional_table(config.experiment_path, "failed_runs")
    experiment_config = read_optional_json(config.experiment_path / "experiment_config.json")
    if config.sort_by not in summary.columns:
        msg = f"sort_by column does not exist in summary: {config.sort_by}"
        raise ReportingError(msg)

    top_runs, schedule_best, cost_best, warnings = build_report_tables(config, summary)
    chart_records = []
    if config.charts:
        chart_records, chart_warnings = generate_charts(
            config=config,
            summary=summary,
            top_runs=top_runs,
        )
        warnings.extend(chart_warnings)

    report_path = config.experiment_path / "report.md"
    report_path.write_text(
        render_markdown_report(
            config=config,
            summary=summary,
            failed_runs=failed_runs,
            top_runs=top_runs,
            best_by_schedule=schedule_best,
            best_by_cost=cost_best,
            experiment_config=experiment_config,
            chart_records=chart_records,
            warnings=warnings,
        ),
        encoding="utf-8",
    )
    write_named_outputs(config.experiment_path / "top_runs", top_runs)
    write_named_outputs(config.experiment_path / "best_by_schedule", schedule_best)
    write_named_outputs(config.experiment_path / "best_by_cost", cost_best)
    return result_payload(config, failed_runs, top_runs, chart_records, warnings, report_path)


def build_report_tables(
    config: IntradayEventReportConfig,
    summary: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[str]]:
    sorted_summary = sort_frame(summary, config.sort_by, config.descending)
    top_runs = ranked_top_runs(sorted_summary, config.top)
    schedule_best, schedule_warnings = best_by_schedule(
        summary,
        config.sort_by,
        config.descending,
    )
    cost_best, cost_warnings = best_by_cost(summary, config.sort_by, config.descending)
    return top_runs, schedule_best, cost_best, [*schedule_warnings, *cost_warnings]


def result_payload(
    config: IntradayEventReportConfig,
    failed_runs: pl.DataFrame,
    top_runs: pl.DataFrame,
    chart_records: list[dict[str, Any]],
    warnings: list[str],
    report_path: Any,
) -> IntradayEventReportResult:
    return IntradayEventReportResult(
        experiment_path=config.experiment_path,
        report_path=report_path,
        top_runs_path=config.experiment_path / "top_runs.csv",
        best_by_schedule_path=config.experiment_path / "best_by_schedule.csv",
        best_by_cost_path=config.experiment_path / "best_by_cost.csv",
        top_run_count=top_runs.height,
        failed_run_count=failed_runs.height,
        best_overall=top_runs.row(0, named=True) if top_runs.height > 0 else None,
        chart_paths=[str(record["path"]) for record in chart_records],
        warnings=warnings,
    )
