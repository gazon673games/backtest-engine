from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from backtest_engine.core.validation import validate_choice as _validate_choice
from backtest_engine.data.instrument_registry import InstrumentSpec, MarginTierSpec
from backtest_engine.data.session_model import SESSION_FILTERS, SHORT_DAY_POLICIES
from backtest_engine.execution.model import EXECUTION_TIMINGS
from backtest_engine.execution.models import EXECUTION_DATA_MODES
from backtest_engine.portfolio.multi.strategy_config import (
    MultiInstrumentStrategyConfig as MultiInstrumentStrategyConfig,
)

MULTI_INSTRUMENT_SIZING_MODES = frozenset({"fixed_contracts", "notional"})
MULTI_INSTRUMENT_RISK_POLICIES = frozenset({"reject", "clip"})
MULTI_INSTRUMENT_CATALOG_ENFORCEMENTS = frozenset({"off", "audit", "strict"})
MULTI_INSTRUMENT_EXECUTION_ENGINES = frozenset({"legacy", "v3"})
MULTI_INSTRUMENT_REBALANCE_POLICIES = frozenset({"independent", "all_or_none"})
MULTI_INSTRUMENT_MARGIN_MODES = frozenset({"cross", "isolated"})
MULTI_INSTRUMENT_UNIVERSE_REMOVAL_POLICIES = frozenset({"liquidate", "hold"})
MULTI_INSTRUMENT_SESSION_FILTERS = SESSION_FILTERS
MULTI_INSTRUMENT_SHORT_DAY_POLICIES = SHORT_DAY_POLICIES


class MultiInstrumentInstrumentSpec(InstrumentSpec):
    pass


class MultiInstrumentLegConfig(BaseModel):
    asset_id: str
    dataset_path: Path | None = None
    catalog_path: Path | None = None
    dataset_id: str | None = None
    catalog_enforcement: str = "off"
    quality_approval_path: Path | None = None
    quality_approval_required: bool = True
    min_usable_decision_day_pct: float = Field(default=0.95, ge=0.0, le=1.0)
    allow_legacy_bar_timestamps: bool = False
    instrument_registry_path: Path | None = None
    calendar_path: Path | None = None
    session_filter: str = "all"
    main_session_start: time = time.min
    main_session_end: time = time.max
    evening_session_start: time | None = None
    evening_session_end: time | None = None
    holiday_dates: list[date] = Field(default_factory=list)
    short_day_dates: list[date] = Field(default_factory=list)
    short_day_policy: str = "allow"
    price_currency: str | None = None
    pnl_currency: str | None = None
    contract_multiplier: float = Field(default=1.0, gt=0.0)
    lot_size: float = Field(default=1.0, gt=0.0)
    tick_size: float | None = Field(default=None, gt=0.0)
    tick_value: float | None = Field(default=None, gt=0.0)
    initial_margin: float | None = Field(default=None, ge=0.0)
    exchange_fee_per_contract: float = Field(default=0.0, ge=0.0)
    asset_class: str = "generic"
    instrument_type: str = "future"
    base_currency: str | None = None
    quote_currency: str | None = None
    settlement_currency: str | None = None
    contract_type: str = "linear"
    quantity_step: float | None = Field(default=None, gt=0.0)
    minimum_quantity: float | None = Field(default=None, ge=0.0)
    minimum_notional: float | None = Field(default=None, ge=0.0)
    maker_fee_bps: float = Field(default=0.0, ge=0.0)
    taker_fee_bps: float = Field(default=0.0, ge=0.0)
    maintenance_margin_rate: float | None = Field(default=None, ge=0.0, lt=1.0)
    liquidation_fee_bps: float = Field(default=0.0, ge=0.0)
    margin_tiers: list[MarginTierSpec] = Field(default_factory=list)
    specs: list[MultiInstrumentInstrumentSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("asset_id")
    @classmethod
    def _validate_asset_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("asset_id must not be empty")
        return normalized

    @field_validator("catalog_enforcement")
    @classmethod
    def _validate_catalog_enforcement(cls, value: str) -> str:
        return _validate_choice(value, MULTI_INSTRUMENT_CATALOG_ENFORCEMENTS, "catalog enforcement")

    @field_validator("session_filter")
    @classmethod
    def _validate_session_filter(cls, value: str) -> str:
        return _validate_choice(value, MULTI_INSTRUMENT_SESSION_FILTERS, "session filter")

    @field_validator("short_day_policy")
    @classmethod
    def _validate_short_day_policy(cls, value: str) -> str:
        return _validate_choice(value, MULTI_INSTRUMENT_SHORT_DAY_POLICIES, "short day policy")

    @model_validator(mode="after")
    def _validate_data_source(self) -> Self:
        if self.dataset_path is None and (self.catalog_path is None or self.dataset_id is None):
            raise ValueError("dataset_path or catalog_path+dataset_id is required")
        return self


class MultiInstrumentUniverseConfig(BaseModel):
    path: Path | None = None
    ranking_column: str = "quote_volume"
    top_n: int | None = Field(default=None, gt=0)
    min_score: float | None = Field(default=None, ge=0.0)
    min_price: float | None = Field(default=None, gt=0.0)
    allowlist: list[str] = Field(default_factory=list)
    blocklist: list[str] = Field(default_factory=list)
    removal_policy: str = "liquidate"
    require_point_in_time: bool = True

    @field_validator("removal_policy")
    @classmethod
    def _validate_removal_policy(cls, value: str) -> str:
        return _validate_choice(
            value,
            MULTI_INSTRUMENT_UNIVERSE_REMOVAL_POLICIES,
            "universe removal policy",
        )

    @model_validator(mode="after")
    def _validate_lists(self) -> Self:
        overlap = set(self.allowlist) & set(self.blocklist)
        if overlap:
            raise ValueError(
                "Universe allowlist and blocklist overlap: " + ", ".join(sorted(overlap))
            )
        return self


class MultiInstrumentPortfolioConfig(BaseModel):
    initial_capital: float = Field(default=1.0, gt=0.0)
    base_currency: str | None = None
    fx_rates: dict[str, float] = Field(default_factory=dict)
    sizing_mode: str = "fixed_contracts"
    fixed_contracts: float = Field(default=1.0, gt=0.0)
    target_notional: float | None = Field(default=None, gt=0.0)
    allow_fractional_contracts: bool = True
    contract_rounding: str = "floor"
    max_position_notional: float | None = Field(default=None, gt=0.0)
    max_leverage: float | None = Field(default=None, gt=0.0)
    max_margin_usage: float | None = Field(default=None, gt=0.0)
    risk_policy: str = "reject"
    collateral_currency: str | None = None
    initial_collateral: float | None = Field(default=None, gt=0.0)
    valuation_rates_path: Path | None = None
    margin_mode: str = "cross"
    leverage: float = Field(default=1.0, ge=1.0)
    require_mark_price: bool = False
    require_margin_tiers: bool = False
    liquidation_target_buffer: float = Field(default=1.05, ge=1.0)
    minimum_liquidation_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    reconciliation_statement_path: Path | None = None
    reconciliation_tolerance: float = Field(default=0.01, ge=0.0)

    @field_validator("sizing_mode")
    @classmethod
    def _validate_sizing_mode(cls, value: str) -> str:
        return _validate_choice(value, MULTI_INSTRUMENT_SIZING_MODES, "sizing mode")

    @field_validator("risk_policy")
    @classmethod
    def _validate_risk_policy(cls, value: str) -> str:
        return _validate_choice(value, MULTI_INSTRUMENT_RISK_POLICIES, "risk policy")

    @field_validator("margin_mode")
    @classmethod
    def _validate_margin_mode(cls, value: str) -> str:
        return _validate_choice(value, MULTI_INSTRUMENT_MARGIN_MODES, "margin mode")

    @model_validator(mode="after")
    def _validate_target_notional(self) -> Self:
        if self.sizing_mode == "notional" and self.target_notional is None:
            raise ValueError("target_notional is required for sizing_mode=notional")
        if self.initial_collateral is not None and self.collateral_currency is None:
            raise ValueError("collateral_currency is required with initial_collateral")
        return self


class MultiInstrumentExecutionConfig(BaseModel):
    partial_fill_policy: Literal["allow", "reject"] = "allow"
    engine: str = "v3"
    timing: str = "signal_close_to_next_bar_close"
    allow_noncausal_same_close: bool = False
    from_date: date = date.min
    till_date: date | None = None
    commission_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)
    slippage_ticks: float = Field(default=0.0, ge=0.0)
    bid_ask_spread_bps: float = Field(default=0.0, ge=0.0)
    bid_ask_spread_ticks: float = Field(default=0.0, ge=0.0)
    volume_participation_limit: float | None = Field(default=None, gt=0.0, le=1.0)
    volume_to_contracts_multiplier: float = Field(default=1.0, gt=0.0)
    latency_bars: int = Field(default=0, ge=0)
    queue_ahead_contracts: float = Field(default=0.0, ge=0.0)
    market_impact_bps: float = Field(default=0.0, ge=0.0)
    rebalance_policy: str = "independent"
    roll_cost_bps: float = Field(default=0.0, ge=0.0)
    roll_slippage_ticks: float = Field(default=0.0, ge=0.0)
    roll_pnl_policy: Literal["reject", "exclude_gap"] = "reject"
    trading_days_per_year: int | None = Field(default=None, gt=0)
    data_mode: str = "bar"
    liquidity_role: str = "taker"
    funding_rate_column: str = "funding_rate"
    default_funding_rate: float = 0.0
    funding_event_column: str = "funding_event"
    require_funding_event_flag: bool = True
    end_of_data_position_policy: Literal["reject", "mark"] = "reject"
    mark_price_column: str = "mark_price"
    mark_low_column: str = "mark_low"
    mark_high_column: str = "mark_high"
    calibration_path: Path | None = None

    @field_validator("timing")
    @classmethod
    def _validate_timing(cls, value: str) -> str:
        return _validate_choice(value, EXECUTION_TIMINGS, "execution timing")

    @field_validator("engine")
    @classmethod
    def _validate_engine(cls, value: str) -> str:
        return _validate_choice(value, MULTI_INSTRUMENT_EXECUTION_ENGINES, "execution engine")

    @field_validator("rebalance_policy")
    @classmethod
    def _validate_rebalance_policy(cls, value: str) -> str:
        return _validate_choice(
            value,
            MULTI_INSTRUMENT_REBALANCE_POLICIES,
            "rebalance policy",
        )

    @field_validator("data_mode")
    @classmethod
    def _validate_data_mode(cls, value: str) -> str:
        return _validate_choice(value, EXECUTION_DATA_MODES, "execution data mode")

    @model_validator(mode="after")
    def _validate_execution_safety(self) -> Self:
        if self.till_date is not None and self.till_date < self.from_date:
            raise ValueError("execution.till_date must be on or after from_date")
        if self.timing == "signal_close_to_same_close" and not self.allow_noncausal_same_close:
            raise ValueError(
                "signal_close_to_same_close is non-causal; set "
                "allow_noncausal_same_close=true only for diagnostic runs"
            )
        if self.latency_bars and self.engine != "v3":
            raise ValueError("latency_bars requires execution.engine=v3")
        return self


class MultiInstrumentOutputConfig(BaseModel):
    root: Path = Path("data/backtests")
    run_id: str


class MultiInstrumentBacktestConfig(BaseModel):
    legs: list[MultiInstrumentLegConfig]
    strategy: MultiInstrumentStrategyConfig
    portfolio: MultiInstrumentPortfolioConfig = Field(
        default_factory=MultiInstrumentPortfolioConfig
    )
    execution: MultiInstrumentExecutionConfig = Field(
        default_factory=MultiInstrumentExecutionConfig
    )
    universe: MultiInstrumentUniverseConfig = Field(default_factory=MultiInstrumentUniverseConfig)
    output: MultiInstrumentOutputConfig

    @model_validator(mode="after")
    def _validate_legs(self) -> Self:
        if not self.legs:
            raise ValueError("legs must not be empty")
        asset_ids = [leg.asset_id for leg in self.legs]
        duplicates = sorted({asset_id for asset_id in asset_ids if asset_ids.count(asset_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate asset_id values: {', '.join(duplicates)}")
        return self


class MultiInstrumentBacktestResult(BaseModel):
    output_dir: Path
    metrics: dict[str, Any]
    dataset_validations: dict[str, dict[str, Any]]
