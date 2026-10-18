from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

ENGINE_PARAM_NAMES = frozenset(
    {
        "decision_time",
        "decision_tolerance_minutes",
        "minimum_bars_since_session_start",
        "commission_bps",
        "slippage_bps",
        "from_date",
    }
)


class IntradayBatchConfig(BaseModel):
    experiment_id: str = Field(min_length=1)
    output_root: Path = Path("data/backtests")
    dataset_path: Path
    quality_approval_path: Path | None = None
    quality_approval_required: bool = True
    min_usable_decision_day_pct: float = Field(default=0.95, ge=0.0, le=1.0)
    allow_legacy_bar_timestamps: bool = False
    strategy_file: Path
    from_date: date = date.min
    decision_time: time | None = None
    decision_tolerance_minutes: int | None = Field(default=None, ge=0)
    minimum_bars_since_session_start: int = Field(default=0, ge=0)
    commission_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)
    output_format_version: int = 1
    grid: dict[str, list[Any]]
    continue_on_error: bool = True


class IntradayBatchRunResult(BaseModel):
    run_id: str
    output_dir: Path
    params: dict[str, Any]
    metrics: dict[str, Any]


class IntradayBatchResult(BaseModel):
    output_dir: Path
    successful_run_count: int
    failed_run_count: int
    summary_rows: list[dict[str, Any]]
    failed_rows: list[dict[str, Any]]
