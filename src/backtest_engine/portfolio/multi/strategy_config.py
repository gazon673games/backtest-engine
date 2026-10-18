"""Pinned SDK sleeves and capital allocation for the native portfolio runtime."""

from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, model_validator


class SDKSleeveConfig(BaseModel):
    asset_id: str
    strategy_file: Path
    source_snapshot: dict[str, Any] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    initial_capital: float = Field(gt=0)
    max_exposure: float = Field(gt=0)
    signal_offset_minutes: int = Field(default=0, ge=0, le=1439)
    # UTC decision boundaries; exits and reductions remain allowed at all times.
    risk_increase_blocked_weekdays: list[Annotated[int, Field(ge=0, le=6)]] = Field(
        default_factory=list
    )
    risk_increase_allowed_hours: list[Annotated[int, Field(ge=0, le=23)]] | None = None


class MultiInstrumentStrategyConfig(BaseModel):
    strategy_file: Path | None = None
    strategy_params: dict[str, str] = Field(default_factory=dict)
    strict_runtime: bool = False
    sdk_sleeves: list[SDKSleeveConfig] = Field(default_factory=list)
    capital_policy: Literal[
        "independent", "monthly_profit", "monthly_rebalance", "monthly_rebalance_total"
    ] = "independent"

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        if (self.strategy_file is None) == (not self.sdk_sleeves):
            raise ValueError("Choose strategy_file or sdk_sleeves")
        ids = [sleeve.asset_id for sleeve in self.sdk_sleeves]
        if len(ids) != len(set(ids)):
            raise ValueError("SDK sleeve asset_id values must be unique")
        if not self.sdk_sleeves and self.capital_policy != "independent":
            raise ValueError("capital_policy requires sdk_sleeves")
        return self
