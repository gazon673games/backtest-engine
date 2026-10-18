from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.reporting.config import ReportConfig
from backtest_engine.reporting.io import write_json
from backtest_engine.reporting.run_charts import (
    plot_drawdown_curve,
    plot_equity_curve,
    plot_parameter_heatmap,
)
from backtest_engine.reporting.tables import unique_strings


def generate_charts(
    *,
    config: ReportConfig,
    summary: pl.DataFrame,
    top_runs: pl.DataFrame,
    best_by_dataset: pl.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    charts_dir = config.experiment_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    chart_metric = chart_metric_name(config, summary)

    if top_runs.height > 0:
        add_best_overall_charts(config, top_runs.row(0, named=True), chart_records, warnings)
    add_dataset_charts(config, best_by_dataset, chart_records, warnings)
    add_heatmaps(config, summary, chart_metric, chart_records, warnings)
    write_chart_manifest(charts_dir, chart_records)
    return chart_records, warnings


def add_best_overall_charts(
    config: ReportConfig,
    best_overall: dict[str, Any],
    chart_records: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    source_run_id = str(best_overall.get("run_id", "best_overall"))
    source_equity_path = equity_path(config.experiment_path, best_overall)
    add_chart(
        chart_records,
        warnings,
        chart_type="equity",
        title="Best overall equity",
        path=config.experiment_path / "charts" / "best_overall_equity.png",
        source_run_id=source_run_id,
        source_metric=config.sort_by,
        continuous_symbol=str(best_overall.get("continuous_symbol", "")),
        create=lambda output_path: plot_equity_curve(
            source_equity_path,
            output_path,
            title="Best overall equity curve",
        ),
    )
    add_chart(
        chart_records,
        warnings,
        chart_type="drawdown",
        title="Best overall drawdown",
        path=config.experiment_path / "charts" / "best_overall_drawdown.png",
        source_run_id=source_run_id,
        source_metric=config.sort_by,
        continuous_symbol=str(best_overall.get("continuous_symbol", "")),
        create=lambda output_path: plot_drawdown_curve(
            source_equity_path,
            output_path,
            title="Best overall drawdown",
        ),
    )


def add_dataset_charts(
    config: ReportConfig,
    best_by_dataset: pl.DataFrame,
    chart_records: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    charts_dir = config.experiment_path / "charts"
    for row in best_by_dataset.iter_rows(named=True):
        symbol = str(row.get("continuous_symbol", "dataset"))
        safe_symbol = safe_filename(symbol)
        source_run_id = str(row.get("run_id", safe_symbol))
        path = equity_path(config.experiment_path, row)
        add_chart(
            chart_records,
            warnings,
            chart_type="equity",
            title=f"{symbol} best equity",
            path=charts_dir / f"{safe_symbol}_best_equity.png",
            source_run_id=source_run_id,
            source_metric=config.sort_by,
            continuous_symbol=symbol,
            create=lambda output_path, symbol=symbol, path=path: plot_equity_curve(
                path,
                output_path,
                title=f"{symbol} best equity curve",
            ),
        )
        add_chart(
            chart_records,
            warnings,
            chart_type="drawdown",
            title=f"{symbol} best drawdown",
            path=charts_dir / f"{safe_symbol}_best_drawdown.png",
            source_run_id=source_run_id,
            source_metric=config.sort_by,
            continuous_symbol=symbol,
            create=lambda output_path, symbol=symbol, path=path: plot_drawdown_curve(
                path,
                output_path,
                title=f"{symbol} best drawdown",
            ),
        )


def add_heatmaps(
    config: ReportConfig,
    summary: pl.DataFrame,
    chart_metric: str | None,
    chart_records: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    if chart_metric is None:
        warnings.append("No numeric chart metric available for heatmaps.")
        return
    charts_dir = config.experiment_path / "charts"
    for symbol in unique_strings(summary, "continuous_symbol"):
        safe_symbol = safe_filename(symbol)
        safe_metric = safe_filename(chart_metric)
        add_chart(
            chart_records,
            warnings,
            chart_type="heatmap",
            title=f"{symbol} {chart_metric} heatmap",
            path=charts_dir / f"{safe_symbol}_{safe_metric}_heatmap.png",
            source_run_id="",
            source_metric=chart_metric,
            continuous_symbol=symbol,
            create=lambda output_path, symbol=symbol: plot_parameter_heatmap(
                summary,
                symbol,
                chart_metric,
                output_path,
            ),
        )


def add_chart(
    chart_records: list[dict[str, Any]],
    warnings: list[str],
    *,
    chart_type: str,
    title: str,
    path: Path,
    source_run_id: str,
    source_metric: str,
    continuous_symbol: str,
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
            "continuous_symbol": continuous_symbol,
        }
    )


def chart_metric_name(config: ReportConfig, summary: pl.DataFrame) -> str | None:
    if config.chart_metric is not None:
        return config.chart_metric if is_numeric_column(summary, config.chart_metric) else None
    if is_numeric_column(summary, config.sort_by):
        return config.sort_by
    return "sharpe" if is_numeric_column(summary, "sharpe") else None


def is_numeric_column(frame: pl.DataFrame, column: str) -> bool:
    dtype = frame.schema.get(column)
    return bool(dtype is not None and dtype.is_numeric())


def equity_path(experiment_path: Path, row: dict[str, Any]) -> Path:
    run_path = Path(str(row["run_path"]))
    if not run_path.is_absolute() and not (run_path / "equity.parquet").exists():
        run_path = experiment_path / run_path
    return run_path / "equity.parquet"


def relative_chart_path(path: Path) -> str:
    return (Path("charts") / path.name).as_posix()


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("._")
    return safe or "chart"


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
