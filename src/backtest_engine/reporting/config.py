from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

OutputFormat = Literal["all", "md", "json", "csv", "parquet"]

TOP_RUN_COLUMNS = [
    "rank",
    "continuous_symbol",
    "adjustment_mode",
    "run_id",
    "run_path",
    "fast_window",
    "slow_window",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "trade_count",
    "win_rate",
    "profit_factor",
    "avg_trade_return",
    "exposure",
]

BEST_RUN_COLUMNS = [
    "continuous_symbol",
    "run_id",
    "fast_window",
    "slow_window",
    "total_return",
    "annualized_return",
    "sharpe",
    "max_drawdown",
    "trade_count",
    "win_rate",
    "profit_factor",
    "run_path",
]

FAILED_COLUMNS = [
    "dataset_path",
    "params",
    "status",
    "error_type",
    "error_message",
]


class ReportConfig(BaseModel):
    experiment_path: Path
    sort_by: str = "sharpe"
    top: int = Field(default=10, gt=0)
    descending: bool = True
    group_by_dataset: bool = True
    output_format: OutputFormat = "all"
    charts: bool = True
    chart_metric: str | None = None


class ReportResult(BaseModel):
    experiment_path: Path
    report_path: Path | None
    top_runs_path: Path | None
    best_by_dataset_path: Path | None
    top_run_count: int
    failed_run_count: int
    best_overall: dict[str, Any] | None
    best_by_dataset: list[dict[str, Any]]
    chart_paths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
