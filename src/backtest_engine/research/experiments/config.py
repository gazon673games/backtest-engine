from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

OutputFormat = Literal["all", "md", "json", "csv", "parquet"]

REGISTRY_COLUMNS = [
    "status",
    "parent_experiment_id",
    "parent_experiment_path",
    "run_id",
    "run_path",
    "manifest_path",
    "manifest_sha256",
    "dataset_path",
    "dataset_sha256",
    "dataset_version_id",
    "dataset_schema_hash",
    "dataset_schema_version",
    "continuous_symbol",
    "timeframe",
    "dataset_type",
    "adjustment_mode",
    "quality_approval_status",
    "catalog_approval_status",
    "strategy_file",
    "strategy_sha256",
    "strategy_params",
    "schedule",
    "decision_time",
    "from_date",
    "execution_timing",
    "missing_decision_policy",
    "commission_bps",
    "slippage_bps",
    "slippage_ticks",
    "bid_ask_spread_bps",
    "bid_ask_spread_ticks",
    "volume_participation_limit",
    "partial_fill_policy",
    "intrabar_ambiguity_policy",
    "sizing_mode",
    "initial_capital",
    "event_count",
    "trade_count",
    "order_count",
    "partial_fill_count",
    "volume_rejected_count",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "net_pnl",
    "final_equity_cash",
    "total_cost_cash",
    "turnover_notional",
    "max_position_notional",
    "exposure",
    "long_exposure",
    "short_exposure",
    "error_type",
    "error_message",
]

TOP_RUN_COLUMNS = [
    "rank",
    "parent_experiment_id",
    "continuous_symbol",
    "timeframe",
    "run_id",
    "run_path",
    "strategy_file",
    "strategy_sha256",
    "strategy_params",
    "schedule",
    "decision_time",
    "total_return",
    "sharpe",
    "max_drawdown",
    "net_pnl",
    "final_equity_cash",
    "trade_count",
    "order_count",
    "total_cost_cash",
    "exposure",
]

BEST_BY_GROUP_COLUMNS = [
    "group_key",
    "parent_experiment_id",
    "continuous_symbol",
    "timeframe",
    "run_id",
    "run_path",
    "strategy_file",
    "strategy_sha256",
    "strategy_params",
    "total_return",
    "sharpe",
    "max_drawdown",
    "net_pnl",
    "trade_count",
    "total_cost_cash",
]


class ExperimentRegistryConfig(BaseModel):
    root: Path
    output_path: Path | None = None
    output_format: OutputFormat = "all"


class ExperimentRegistryResult(BaseModel):
    root: Path
    output_path: Path | None
    registry_path: Path | None
    run_count: int
    successful_run_count: int
    failed_run_count: int
    experiment_count: int
    warnings: list[str] = Field(default_factory=list)


class ExperimentCompareConfig(BaseModel):
    registry_path: Path | None = None
    root: Path | None = None
    output_path: Path = Path("data/backtests/experiment_compare")
    sort_by: str = "sharpe"
    top: int = Field(default=20, gt=0)
    descending: bool = True
    group_by: list[str] = Field(default_factory=list)
    output_format: OutputFormat = "all"

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        if self.registry_path is None and self.root is None:
            raise ValueError("registry_path or root is required")
        return self


class ExperimentCompareResult(BaseModel):
    output_path: Path
    report_path: Path | None
    registry_path: Path | None
    top_runs_path: Path | None
    best_by_group_path: Path | None
    run_count: int
    successful_run_count: int
    failed_run_count: int
    top_run_count: int
    best_overall: dict[str, Any] | None
    warnings: list[str] = Field(default_factory=list)
