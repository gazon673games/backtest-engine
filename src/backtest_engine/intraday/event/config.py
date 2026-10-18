from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from backtest_engine.core.validation import validate_choice as _validate_choice
from backtest_engine.data.instrument_registry import InstrumentSpec, MarginTierSpec
from backtest_engine.data.session_model import SESSION_FILTERS, SHORT_DAY_POLICIES
from backtest_engine.execution.model import (
    EXECUTION_INTRABAR_AMBIGUITY_POLICIES,
    EXECUTION_ORDER_TYPES,
    EXECUTION_TIMINGS,
)
from backtest_engine.execution.models import EXECUTION_DATA_MODES
from backtest_engine.intraday.event.result_models import (
    IntradayEventBacktestResult as IntradayEventBacktestResult,
)

INTRADAY_EVENT_SCHEDULES = frozenset({"every-bar", "times"})
INTRADAY_EVENT_MISSING_DECISION_POLICIES = frozenset({"carry", "close", "skip"})
INTRADAY_EVENT_TIMINGS = EXECUTION_TIMINGS
INTRADAY_EVENT_SIZING_MODES = frozenset(
    {"signal_units", "fixed_contracts", "notional", "strategy_contracts"}
)
INTRADAY_EVENT_CONTRACT_ROUNDING = frozenset({"floor", "round", "ceil"})
INTRADAY_EVENT_RISK_POLICIES = frozenset({"reject", "clip"})
INTRADAY_EVENT_SESSION_FILTERS = SESSION_FILTERS
INTRADAY_EVENT_SHORT_DAY_POLICIES = SHORT_DAY_POLICIES
INTRADAY_EVENT_ORDER_TYPES = EXECUTION_ORDER_TYPES
INTRADAY_EVENT_EXECUTION_ENGINES = frozenset({"legacy", "v3"})
INTRADAY_EVENT_PARTIAL_FILL_POLICIES = frozenset({"allow", "reject"})
INTRADAY_EVENT_INTRABAR_AMBIGUITY_POLICIES = EXECUTION_INTRABAR_AMBIGUITY_POLICIES
INTRADAY_EVENT_CATALOG_ENFORCEMENTS = frozenset({"off", "audit", "strict"})
INTRADAY_EVENT_LIQUIDITY_ROLES = frozenset({"maker", "taker"})
INTRADAY_EVENT_MARGIN_MODES = frozenset({"cross", "isolated"})


class IntradayEventDatasetConfig(BaseModel):
    path: Path
    catalog_path: Path | None = None
    dataset_id: str | None = None
    catalog_enforcement: str = "off"
    quality_approval_path: Path | None = None
    quality_approval_required: bool = True
    min_usable_decision_day_pct: float = Field(default=0.95, ge=0.0, le=1.0)
    allow_legacy_bar_timestamps: bool = False

    @field_validator("catalog_enforcement")
    @classmethod
    def _validate_catalog_enforcement(cls, value: str) -> str:
        return _validate_choice(
            value,
            INTRADAY_EVENT_CATALOG_ENFORCEMENTS,
            "catalog enforcement",
        )


class IntradayEventStrategyConfig(BaseModel):
    strategy_file: Path
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    strict_runtime: bool = False
    timeframe: str | None = None


class IntradayEventInstrumentSpec(InstrumentSpec):
    pass


class IntradayEventInstrumentConfig(BaseModel):
    spec_path: Path | None = None
    registry_path: Path | None = None
    symbol: str | None = None
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
    specs: list[IntradayEventInstrumentSpec] = Field(default_factory=list)


class IntradayEventPortfolioConfig(BaseModel):
    initial_capital: float = Field(default=1.0, gt=0.0)
    base_currency: str | None = None
    fx_rates: dict[str, float] = Field(default_factory=dict)
    sizing_mode: str = "signal_units"
    fixed_contracts: float = Field(default=1.0, gt=0.0)
    target_notional: float | None = Field(default=None, gt=0.0)
    contract_rounding: str = "floor"
    allow_fractional_contracts: bool = True
    max_position_notional: float | None = Field(default=None, gt=0.0)
    max_leverage: float | None = Field(default=None, gt=0.0)
    max_margin_usage: float | None = Field(default=None, gt=0.0)
    risk_policy: str = "reject"
    collateral_currency: str | None = None
    margin_mode: str = "cross"
    leverage: float = Field(default=1.0, ge=1.0)
    valuation_rates_path: Path | None = None
    initial_collateral: float | None = Field(default=None, gt=0.0)
    require_mark_price: bool = False
    require_margin_tiers: bool = False
    liquidation_target_buffer: float = Field(default=1.05, ge=1.0)
    minimum_liquidation_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    reconciliation_statement_path: Path | None = None
    reconciliation_tolerance: float = Field(default=0.01, ge=0.0)

    @field_validator("sizing_mode")
    @classmethod
    def _validate_sizing_mode(cls, value: str) -> str:
        return _validate_choice(value, INTRADAY_EVENT_SIZING_MODES, "sizing mode")

    @field_validator("contract_rounding")
    @classmethod
    def _validate_contract_rounding(cls, value: str) -> str:
        return _validate_choice(
            value,
            INTRADAY_EVENT_CONTRACT_ROUNDING,
            "contract rounding",
        )

    @field_validator("risk_policy")
    @classmethod
    def _validate_risk_policy(cls, value: str) -> str:
        return _validate_choice(value, INTRADAY_EVENT_RISK_POLICIES, "risk policy")

    @field_validator("margin_mode")
    @classmethod
    def _validate_margin_mode(cls, value: str) -> str:
        return _validate_choice(value, INTRADAY_EVENT_MARGIN_MODES, "margin mode")

    @model_validator(mode="after")
    def _validate_target_notional(self) -> Self:
        if self.sizing_mode == "notional" and self.target_notional is None:
            raise ValueError("target_notional is required for sizing_mode=notional")
        if self.initial_collateral is not None and self.collateral_currency is None:
            raise ValueError("collateral_currency is required with initial_collateral")
        return self


class IntradayEventSessionConfig(BaseModel):
    calendar_path: Path | None = None
    trading_calendar: str = "observed"
    holiday_dates: list[date] = Field(default_factory=list)
    short_day_dates: list[date] = Field(default_factory=list)
    session_filter: str = "all"
    main_session_start: time = time.min
    main_session_end: time = time.max
    evening_session_start: time | None = None
    evening_session_end: time | None = None
    short_day_policy: str = "allow"

    @field_validator("session_filter")
    @classmethod
    def _validate_session_filter(cls, value: str) -> str:
        return _validate_choice(value, INTRADAY_EVENT_SESSION_FILTERS, "session filter")

    @field_validator("short_day_policy")
    @classmethod
    def _validate_short_day_policy(cls, value: str) -> str:
        return _validate_choice(value, INTRADAY_EVENT_SHORT_DAY_POLICIES, "short day policy")


class IntradayEventExecutionConfig(BaseModel):
    engine: str = "v3"
    timing: str = "signal_close_to_next_bar_close"
    allow_noncausal_same_close: bool = False
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
    latency_bars: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    order_book_path: Path | None = None
    l2_queue_ahead_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    queue_ahead_contracts: float = Field(default=0.0, ge=0.0)
    market_impact_bps: float = Field(default=0.0, ge=0.0)
    partial_fill_policy: str = "allow"
    intrabar_ambiguity_policy: str = "conservative"
    roll_cost_bps: float = Field(default=0.0, ge=0.0)
    roll_slippage_ticks: float = Field(default=0.0, ge=0.0)
    roll_pnl_policy: Literal["reject", "exclude_gap"] = "reject"
    trading_days_per_year: int | None = Field(default=None, gt=0)
    liquidity_role: str = "taker"
    funding_rate_column: str = "funding_rate"
    default_funding_rate: float = 0.0
    funding_event_column: str = "funding_event"
    require_funding_event_flag: bool = True
    end_of_data_position_policy: Literal["reject", "mark"] = "reject"
    data_mode: str = "bar"
    mark_price_column: str = "mark_price"
    mark_low_column: str = "mark_low"
    mark_high_column: str = "mark_high"
    calibration_path: Path | None = None

    @field_validator("engine")
    @classmethod
    def _validate_engine(cls, value: str) -> str:
        return _validate_choice(value, INTRADAY_EVENT_EXECUTION_ENGINES, "execution engine")

    @field_validator("timing")
    @classmethod
    def _validate_timing(cls, value: str) -> str:
        return _validate_choice(value, INTRADAY_EVENT_TIMINGS, "intraday event timing")

    @field_validator("schedule")
    @classmethod
    def _validate_schedule(cls, value: str) -> str:
        return _validate_choice(value, INTRADAY_EVENT_SCHEDULES, "intraday event schedule")

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
        return _validate_choice(
            value,
            INTRADAY_EVENT_PARTIAL_FILL_POLICIES,
            "partial fill policy",
        )

    @field_validator("intrabar_ambiguity_policy")
    @classmethod
    def _validate_intrabar_ambiguity_policy(cls, value: str) -> str:
        return _validate_choice(
            value,
            INTRADAY_EVENT_INTRABAR_AMBIGUITY_POLICIES,
            "intrabar ambiguity policy",
        )

    @field_validator("liquidity_role")
    @classmethod
    def _validate_liquidity_role(cls, value: str) -> str:
        return _validate_choice(value, INTRADAY_EVENT_LIQUIDITY_ROLES, "liquidity role")

    @field_validator("data_mode")
    @classmethod
    def _validate_data_mode(cls, value: str) -> str:
        return _validate_choice(value, EXECUTION_DATA_MODES, "execution data mode")

    @model_validator(mode="after")
    def _validate_decision_times(self) -> Self:
        if self.schedule == "times" and not self.decision_times:
            raise ValueError("decision_times must contain at least one time for schedule=times")
        if self.timing == "signal_close_to_same_close" and not self.allow_noncausal_same_close:
            raise ValueError(
                "signal_close_to_same_close exposes the current close to the strategy and "
                "fills at that same close; set allow_noncausal_same_close=true only for "
                "explicit diagnostic runs"
            )
        return self


class IntradayEventOutputConfig(BaseModel):
    root: Path = Path("data/backtests")
    run_id: str = Field(min_length=1)
    write_csv: bool = False


class IntradayEventBacktestConfig(BaseModel):
    dataset: IntradayEventDatasetConfig
    strategy: IntradayEventStrategyConfig
    instrument: IntradayEventInstrumentConfig = Field(default_factory=IntradayEventInstrumentConfig)
    portfolio: IntradayEventPortfolioConfig = Field(default_factory=IntradayEventPortfolioConfig)
    session: IntradayEventSessionConfig = Field(default_factory=IntradayEventSessionConfig)
    execution: IntradayEventExecutionConfig = Field(default_factory=IntradayEventExecutionConfig)
    output: IntradayEventOutputConfig
