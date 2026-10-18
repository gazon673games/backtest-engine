from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from backtest_engine.accounting.funding_policy import FundingConfig


class WorkbenchRunRequest(FundingConfig):
    dataset_path: Path
    quality_approval_path: Path | None = None
    quality_approval_required: bool = True
    allow_legacy_bar_timestamps: bool = False
    script_path: Path | None = None
    script_source: str | None = None
    script_language: Literal["python", "javascript"] = "javascript"
    params: dict[str, Any] = Field(default_factory=dict)
    from_date: str = "0001-01-01"
    decision_times: list[str] = Field(default_factory=list)
    decision_tolerance_minutes: int | None = Field(default=None, ge=0)
    output_root: Path = Path("data/workbench/runs")
    run_id: str | None = None
    schedule: str = "every-bar"
    missing_decision_policy: str = "carry"
    execution_timing: str = "signal_close_to_next_bar_close"
    allow_noncausal_same_close: bool = False
    end_of_data_position_policy: Literal["reject", "mark"] = "mark"
    strict_runtime: bool = True
    chart_limit: int = Field(default=5000, ge=100)
    sizing_mode: str = "strategy_contracts"
    initial_capital: float = Field(default=1.0, gt=0.0)
    base_currency: str | None = None
    fx_rates: dict[str, float] = Field(default_factory=dict)
    valuation_rates_path: Path | None = None
    collateral_currency: str | None = None
    initial_collateral: float | None = Field(default=None, gt=0.0)
    margin_mode: Literal["cross", "isolated"] = "cross"
    leverage: float = Field(default=1.0, ge=1.0)
    trading_days_per_year: int | None = Field(default=None, gt=0)
    liquidity_role: Literal["maker", "taker"] = "taker"
    fixed_contracts: float = Field(default=1.0, gt=0.0)
    target_notional: float | None = Field(default=None, gt=0.0)
    contract_rounding: str = "floor"
    allow_fractional_contracts: bool | None = None
    max_position_notional: float | None = Field(default=None, gt=0.0)
    max_leverage: float | None = Field(default=None, gt=0.0)
    max_margin_usage: float | None = Field(default=None, gt=0.0)
    risk_policy: str = "reject"
    contract_multiplier: float = Field(default=1.0, gt=0.0)
    lot_size: float = Field(default=1.0, gt=0.0)
    tick_size: float | None = Field(default=None, gt=0.0)
    tick_value: float | None = Field(default=None, gt=0.0)
    exchange_fee_per_contract: float = Field(default=0.0, ge=0.0)
    instrument_registry_path: Path | None = None
    calendar_path: Path | None = None
    catalog_path: Path | None = None
    dataset_id: str | None = None
    catalog_enforcement: str = "off"
    session_filter: str = "all"
    commission_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)
    slippage_ticks: float = Field(default=0.0, ge=0.0)
    bid_ask_spread_bps: float = Field(default=0.0, ge=0.0)
    bid_ask_spread_ticks: float = Field(default=0.0, ge=0.0)
    execution_engine: str = "v3"
    volume_participation_limit: float | None = Field(default=0.1, gt=0.0, le=1.0)
    volume_to_contracts_multiplier: float = Field(default=1.0, gt=0.0)
    latency_bars: int = Field(default=0, ge=0)
    queue_ahead_contracts: float = Field(default=0.0, ge=0.0)
    market_impact_bps: float = Field(default=0.0, ge=0.0)
    partial_fill_policy: str = "allow"
    intrabar_ambiguity_policy: str = "conservative"
    roll_cost_bps: float = Field(default=0.0, ge=0.0)
    roll_slippage_ticks: float = Field(default=0.0, ge=0.0)
    roll_pnl_policy: Literal["reject", "exclude_gap"] = "exclude_gap"

    @model_validator(mode="after")
    def _validate_script(self) -> WorkbenchRunRequest:
        if self.script_path is None and not self.script_source:
            raise ValueError("script_path or script_source is required")
        if self.script_path is not None and self.script_source:
            raise ValueError("Use either script_path or script_source, not both")
        return self


class WorkbenchRunResponse(BaseModel):
    status: str
    run_id: str
    output_dir: Path | None = None
    report: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
