from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from backtest_engine.data.instrument_registry import InstrumentSpec
from backtest_engine.data.market_identity import account_currency
from backtest_engine.execution.models import EXECUTION_DATA_MODES


class PaperTradingConfig(BaseModel):
    strategy_file: Path
    strategy_params: dict[str, str] = Field(default_factory=dict)
    instrument: InstrumentSpec
    initial_capital: float = Field(default=10_000.0, gt=0.0)
    base_currency: str | None = None
    collateral_currency: str | None = None
    fixed_contracts: float = Field(default=1.0, gt=0.0)
    leverage: float = Field(default=1.0, ge=1.0)
    liquidity_role: str = "taker"
    data_mode: str = "bar"
    volume_participation_limit: float | None = Field(default=None, gt=0.0, le=1.0)
    latency_events: int = Field(default=0, ge=0)
    queue_ahead_quantity: float = Field(default=0.0, ge=0.0)
    market_impact_bps: float = Field(default=0.0, ge=0.0)
    checkpoint_path: Path | None = None
    checkpoint_every_events: int = Field(default=100, ge=1)
    max_history_bars: int = Field(default=10_000, ge=1)
    max_state_rows: int = Field(default=10_000, ge=1)
    strict_runtime: bool = True
    require_mark_price: bool = True
    require_margin_tiers: bool = True

    @model_validator(mode="after")
    def resolve_currency(self) -> PaperTradingConfig:
        self.base_currency = account_currency(
            self.base_currency,
            [
                self.instrument.pnl_currency
                or self.instrument.settlement_currency
                or self.instrument.quote_currency
                or self.instrument.price_currency,
            ],
        )
        return self

    @field_validator("data_mode")
    @classmethod
    def _validate_data_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in EXECUTION_DATA_MODES:
            raise ValueError(f"data_mode must be one of: {', '.join(sorted(EXECUTION_DATA_MODES))}")
        return normalized

    @field_validator("liquidity_role")
    @classmethod
    def _validate_liquidity_role(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"maker", "taker"}:
            raise ValueError("liquidity_role must be maker or taker")
        return normalized


class PaperTradingResult(BaseModel):
    event_count: int
    fill_count: int
    final_position: float
    final_equity_cash: float
    state_rows: list[dict[str, Any]] = Field(default_factory=list)
    checkpoint_path: Path | None = None
