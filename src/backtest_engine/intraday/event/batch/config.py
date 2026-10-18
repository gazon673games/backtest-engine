from __future__ import annotations

import os
from datetime import date, time
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from backtest_engine.intraday.event.engine import (
    INTRADAY_EVENT_CATALOG_ENFORCEMENTS,
    INTRADAY_EVENT_INTRABAR_AMBIGUITY_POLICIES,
    INTRADAY_EVENT_MISSING_DECISION_POLICIES,
    INTRADAY_EVENT_PARTIAL_FILL_POLICIES,
    INTRADAY_EVENT_SCHEDULES,
    INTRADAY_EVENT_TIMINGS,
    IntradayEventInstrumentConfig,
    IntradayEventPortfolioConfig,
    IntradayEventSessionConfig,
)

ENGINE_PARAM_NAMES = frozenset(
    {
        "schedule",
        "timing",
        "decision_time",
        "decision_tolerance_minutes",
        "missing_decision_policy",
        "from_date",
        "commission_bps",
        "slippage_bps",
        "slippage_ticks",
        "bid_ask_spread_bps",
        "bid_ask_spread_ticks",
        "volume_participation_limit",
        "volume_to_contracts_multiplier",
        "partial_fill_policy",
        "intrabar_ambiguity_policy",
        "roll_cost_bps",
        "roll_slippage_ticks",
        "roll_pnl_policy",
    }
)


def default_max_workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


class IntradayEventBatchConfig(BaseModel):
    experiment_id: str = Field(min_length=1)
    output_root: Path = Path("data/backtests")
    dataset_paths: list[Path] = Field(default_factory=list)
    catalog_path: Path | None = None
    dataset_ids: list[str] = Field(default_factory=list)
    catalog_enforcement: str = "off"
    quality_approval_path: Path | None = None
    quality_approval_required: bool = True
    min_usable_decision_day_pct: float = Field(default=0.95, ge=0.0, le=1.0)
    allow_legacy_bar_timestamps: bool = False
    strategy_file: Path
    instrument: IntradayEventInstrumentConfig = Field(default_factory=IntradayEventInstrumentConfig)
    portfolio: IntradayEventPortfolioConfig = Field(default_factory=IntradayEventPortfolioConfig)
    session: IntradayEventSessionConfig = Field(default_factory=IntradayEventSessionConfig)
    timing: str = "signal_close_to_next_bar_close"
    allow_noncausal_same_close: bool = False
    end_of_data_position_policy: Literal["reject", "mark"] = "reject"
    from_date: date = date.min
    schedule: str = "every-bar"
    decision_times: list[time] = Field(default_factory=list)
    decision_tolerance_minutes: int | None = Field(default=None, ge=0)
    missing_decision_policy: str = "carry"
    commission_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)
    slippage_ticks: float = Field(default=0.0, ge=0.0)
    bid_ask_spread_bps: float = Field(default=0.0, ge=0.0)
    bid_ask_spread_ticks: float = Field(default=0.0, ge=0.0)
    volume_participation_limit: float | None = Field(default=None, gt=0.0, le=1.0)
    volume_to_contracts_multiplier: float = Field(default=1.0, gt=0.0)
    partial_fill_policy: str = "allow"
    intrabar_ambiguity_policy: str = "conservative"
    roll_cost_bps: float = Field(default=0.0, ge=0.0)
    roll_slippage_ticks: float = Field(default=0.0, ge=0.0)
    roll_pnl_policy: Literal["reject", "exclude_gap"] = "reject"
    output_format_version: int = 1
    grid: dict[str, list[Any]]
    continue_on_error: bool = True
    max_workers: int = Field(default_factory=default_max_workers, ge=1, le=64)
    resume_completed: bool = True
    verify_cached_output_hashes: bool = True

    @model_validator(mode="after")
    def _validate_dataset_source(self) -> Self:
        if not self.dataset_paths and not (self.catalog_path and self.dataset_ids):
            raise ValueError("dataset_paths or catalog_path+dataset_ids is required")
        return self

    @field_validator("catalog_enforcement")
    @classmethod
    def _validate_catalog_enforcement(cls, value: str) -> str:
        return _validate_choice(value, INTRADAY_EVENT_CATALOG_ENFORCEMENTS, "catalog enforcement")

    @field_validator("schedule")
    @classmethod
    def _validate_schedule(cls, value: str) -> str:
        return _validate_choice(value, INTRADAY_EVENT_SCHEDULES, "intraday event batch schedule")

    @field_validator("timing")
    @classmethod
    def _validate_timing(cls, value: str) -> str:
        return _validate_choice(value, INTRADAY_EVENT_TIMINGS, "intraday event timing")

    @field_validator("missing_decision_policy")
    @classmethod
    def _validate_missing_decision_policy(cls, value: str) -> str:
        return _validate_choice(
            value,
            INTRADAY_EVENT_MISSING_DECISION_POLICIES,
            "missing decision policy",
        )

    @field_validator("partial_fill_policy")
    @classmethod
    def _validate_partial_fill_policy(cls, value: str) -> str:
        return _validate_choice(value, INTRADAY_EVENT_PARTIAL_FILL_POLICIES, "partial fill policy")

    @field_validator("intrabar_ambiguity_policy")
    @classmethod
    def _validate_intrabar_ambiguity_policy(cls, value: str) -> str:
        return _validate_choice(
            value,
            INTRADAY_EVENT_INTRABAR_AMBIGUITY_POLICIES,
            "intrabar ambiguity policy",
        )


class IntradayEventBatchResult(BaseModel):
    output_dir: Path
    successful_run_count: int
    failed_run_count: int
    resumed_run_count: int = 0
    skipped_duplicate_count: int = 0
    summary_rows: list[dict[str, Any]]
    failed_rows: list[dict[str, Any]]


def _validate_choice(value: str, choices: frozenset[str], label: str) -> str:
    normalized = value.strip().lower()
    if normalized not in choices:
        msg = f"Invalid {label}: {value}. Expected one of: {', '.join(sorted(choices))}"
        raise ValueError(msg)
    return normalized
