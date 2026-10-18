from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field


class ResearchValidationConfig(BaseModel):
    experiment_path: Path
    output_path: Path | None = None
    split_date: date | None = None
    rolling_folds: int = Field(default=3, ge=1)
    rolling_min_train_dates: int = Field(default=20, ge=2)
    rolling_train_dates: int | None = Field(default=None, ge=2)
    purge_dates: int = Field(default=0, ge=0)
    embargo_dates: int = Field(default=0, ge=0)
    selection_metric: str = "total_return"
    locked_holdout_start: date | None = None
    locked_holdout_fraction: float = Field(default=0.2, gt=0.0, lt=0.5)
    group_columns: list[str] = Field(default_factory=list)
    bootstrap_iterations: int = Field(default=200, ge=0)
    bootstrap_seed: int = 1
    block_bootstrap_iterations: int = Field(default=100, ge=0)
    block_bootstrap_length: int = Field(default=5, gt=0)
    cost_stress_bps: list[float] = Field(default_factory=lambda: [1.0, 2.0, 5.0])
    max_parameter_runs_without_warning: int = Field(default=50, gt=0)
    min_best_oos_to_is_return_ratio: float = Field(default=0.5, ge=0.0)
    min_best_oos_total_return: float = 0.0
    gate: bool = False
    max_negative_oos_run_share: float = Field(default=0.5, ge=0.0, le=1.0)
    max_warning_count: int = Field(default=0, ge=0)
    selection_bias_bootstrap_iterations: int = Field(default=500, ge=0)
    selection_bias_seed: int = 1
    pbo_partitions: int = Field(default=8, ge=4)
    max_probability_of_backtest_overfitting: float = Field(default=0.5, ge=0.0, le=1.0)
    max_reality_check_p_value: float = Field(default=0.1, ge=0.0, le=1.0)
    min_deflated_sharpe_probability: float = Field(default=0.95, ge=0.0, le=1.0)


class ResearchValidationResult(BaseModel):
    output_path: Path
    walk_forward_path: Path
    rolling_walk_forward_path: Path
    locked_holdout_path: Path
    robustness_path: Path
    guardrails_path: Path
    gate_path: Path
    bootstrap_path: Path
    block_bootstrap_path: Path
    cost_stress_path: Path
    regime_path: Path
    selection_bias_path: Path
    deflated_sharpe_path: Path
    report_path: Path
    manifest_path: Path
    run_count: int
    warning_count: int
    acceptance_status: str = "not_evaluated"
    warnings: list[str] = Field(default_factory=list)


DEFAULT_GROUP_COLUMNS = (
    "decision_time",
    "window_minutes",
    "window_bars",
    "min_session_bars",
    "max_session_bars",
    "filter_minutes",
    "short_filter",
)
