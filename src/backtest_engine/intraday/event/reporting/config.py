from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

TOP_RUN_COLUMNS = [
    "rank",
    "continuous_symbol",
    "timeframe",
    "run_id",
    "run_path",
    "schedule",
    "decision_time",
    "window_minutes",
    "window_bars",
    "commission_bps",
    "slippage_bps",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "trade_count",
    "long_trade_count",
    "short_trade_count",
    "win_rate",
    "profit_factor",
    "avg_trade_return",
    "exposure",
    "long_exposure",
    "short_exposure",
]

BEST_BY_SCHEDULE_COLUMNS = [
    "schedule",
    "decision_time",
    "run_id",
    "window_minutes",
    "window_bars",
    "commission_bps",
    "slippage_bps",
    "total_return",
    "sharpe",
    "max_drawdown",
    "trade_count",
    "run_path",
]

BEST_BY_COST_COLUMNS = [
    "cost_level_bps",
    "commission_bps",
    "slippage_bps",
    "run_id",
    "decision_time",
    "window_minutes",
    "window_bars",
    "total_return",
    "sharpe",
    "max_drawdown",
    "trade_count",
    "run_path",
]

FAILED_COLUMNS = [
    "dataset_path",
    "params",
    "status",
    "error_type",
    "error_message",
]


class IntradayEventReportConfig(BaseModel):
    experiment_path: Path
    sort_by: str = "sharpe"
    top: int = Field(default=20, gt=0)
    descending: bool = True
    charts: bool = True


class IntradayEventReportResult(BaseModel):
    experiment_path: Path
    report_path: Path
    top_runs_path: Path
    best_by_schedule_path: Path | None
    best_by_cost_path: Path | None
    top_run_count: int
    failed_run_count: int
    best_overall: dict[str, Any] | None
    chart_paths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
