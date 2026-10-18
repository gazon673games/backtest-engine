from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.intraday.event.reporting.config import IntradayEventReportConfig
from backtest_engine.intraday.event.reporting.io import write_json
from backtest_engine.intraday.event.reporting.plotting import (
    plot_cost_stress,
    plot_decision_window_heatmap,
    plot_long_short_exposure,
)
from backtest_engine.reporting.run_charts import plot_drawdown_curve, plot_equity_curve


def generate_charts(
    *,
    config: IntradayEventReportConfig,
    summary: pl.DataFrame,
    top_runs: pl.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    charts_dir = config.experiment_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    add_best_overall_charts(config, top_runs, chart_records, warnings)
    add_summary_charts(config, summary, top_runs, chart_records, warnings)
    write_chart_manifest(charts_dir, chart_records)
    return chart_records, warnings


def add_best_overall_charts(
    config: IntradayEventReportConfig,
    top_runs: pl.DataFrame,
    chart_records: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    if top_runs.height == 0:
        return
    best_overall = top_runs.row(0, named=True)
    source_run_id = str(best_overall.get("run_id", "best_overall"))
    source_equity_path = equity_path(config.experiment_path, best_overall)
    try_chart(
        chart_records,
        warnings,
        chart_type="equity",
        title="Best overall equity",
        path=config.experiment_path / "charts" / "best_overall_equity.png",
        source_run_id=source_run_id,
        source_metric=config.sort_by,
        create=lambda output_path: plot_equity_curve(
            source_equity_path,
            output_path,
            title="Best overall intraday event equity",
        ),
    )
    try_chart(
        chart_records,
        warnings,
        chart_type="drawdown",
        title="Best overall drawdown",
        path=config.experiment_path / "charts" / "best_overall_drawdown.png",
        source_run_id=source_run_id,
        source_metric=config.sort_by,
        create=lambda output_path: plot_drawdown_curve(
            source_equity_path,
            output_path,
            title="Best overall intraday event drawdown",
        ),
    )


def add_summary_charts(
    config: IntradayEventReportConfig,
    summary: pl.DataFrame,
    top_runs: pl.DataFrame,
    chart_records: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    charts_dir = config.experiment_path / "charts"
    chart_specs = [
        (
            "heatmap",
            "Sharpe by decision time and window",
            "sharpe_by_decision_time_window_minutes.png",
            "sharpe",
            lambda path: plot_decision_window_heatmap(
                summary,
                "sharpe",
                path,
                title="Sharpe by decision time and window",
            ),
        ),
        (
            "heatmap",
            "Total return by decision time and window",
            "total_return_by_decision_time_window_minutes.png",
            "total_return",
            lambda path: plot_decision_window_heatmap(
                summary,
                "total_return",
                path,
                title="Total return by decision time and window",
            ),
        ),
        (
            "cost_stress",
            "Cost stress Sharpe",
            "cost_stress_sharpe.png",
            "sharpe",
            lambda path: plot_cost_stress(summary, "sharpe", path, title="Cost stress: Sharpe"),
        ),
        (
            "cost_stress",
            "Cost stress total return",
            "cost_stress_total_return.png",
            "total_return",
            lambda path: plot_cost_stress(
                summary,
                "total_return",
                path,
                title="Cost stress: total return",
            ),
        ),
        (
            "exposure",
            "Long/short exposure top runs",
            "long_short_exposure_top_runs.png",
            config.sort_by,
            lambda path: plot_long_short_exposure(
                top_runs,
                path,
                title="Long/short exposure in top runs",
            ),
        ),
    ]
    for chart_type, title, file_name, source_metric, create in chart_specs:
        try_chart(
            chart_records,
            warnings,
            chart_type=chart_type,
            title=title,
            path=charts_dir / file_name,
            source_run_id="",
            source_metric=source_metric,
            create=create,
        )


def try_chart(
    chart_records: list[dict[str, Any]],
    warnings: list[str],
    *,
    chart_type: str,
    title: str,
    path: Path,
    source_run_id: str,
    source_metric: str,
    create: Any,
) -> None:
    try:
        created_path = create(path)
    except Exception as exc:
        warnings.append(f"Skipped {title}: {exc}")
        return
    chart_records.append(
        {
            "chart_type": chart_type,
            "title": title,
            "path": relative_chart_path(created_path),
            "source_run_id": source_run_id,
            "source_metric": source_metric,
        }
    )


def equity_path(experiment_path: Path, row: dict[str, Any]) -> Path:
    run_path = Path(str(row["run_path"]))
    if not run_path.is_absolute() and not (run_path / "equity.parquet").exists():
        run_path = experiment_path / run_path
    return run_path / "equity.parquet"


def relative_chart_path(path: Path) -> str:
    return (Path("charts") / path.name).as_posix()


def write_chart_manifest(charts_dir: Path, chart_records: list[dict[str, Any]]) -> None:
    write_json(
        charts_dir / "chart_manifest.json",
        [
            {
                **record,
                "path": str(Path(str(record["path"])).as_posix()),
            }
            for record in chart_records
        ],
    )
