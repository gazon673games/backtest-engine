from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class IntradayDatasetConfig(BaseModel):
    path: Path
    quality_approval_path: Path | None = None
    quality_approval_required: bool = True
    min_usable_decision_day_pct: float = Field(default=0.95, ge=0.0, le=1.0)
    allow_legacy_bar_timestamps: bool = False


class IntradayStrategyConfig(BaseModel):
    strategy_file: Path
    strategy_params: dict[str, str] = Field(default_factory=dict)


class IntradayExecutionConfig(BaseModel):
    timing: str = "decision_close_to_next_decision_close"
    from_date: date = date.min
    decision_time: time | None = None
    decision_tolerance_minutes: int | None = Field(default=None, ge=0)
    minimum_bars_since_session_start: int = Field(default=0, ge=0)
    commission_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)
    trading_days_per_year: int | None = Field(default=None, gt=0)


class IntradayOutputConfig(BaseModel):
    root: Path = Path("data/backtests")
    run_id: str = Field(min_length=1)
    write_csv: bool = False


class IntradayBacktestConfig(BaseModel):
    dataset: IntradayDatasetConfig
    strategy: IntradayStrategyConfig
    execution: IntradayExecutionConfig = Field(default_factory=IntradayExecutionConfig)
    output: IntradayOutputConfig


class IntradayBacktestResult(BaseModel):
    output_dir: Path
    metrics: dict[str, Any]
    dataset_validation: dict[str, Any]
    quality_approval: dict[str, Any]
