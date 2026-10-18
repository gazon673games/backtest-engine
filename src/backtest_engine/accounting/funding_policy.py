"""Explicit funding assumptions with backwards-compatible legacy defaults."""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

FundingMode = Literal["legacy", "disabled", "historical"]


class FundingConfig(BaseModel):
    funding_mode: FundingMode = "legacy"
    funding_rate_column: str = "funding_rate"
    default_funding_rate: float = Field(default=0.0, allow_inf_nan=False)
    funding_event_column: str = "funding_event"
    require_funding_event_flag: bool = True

    @model_validator(mode="after")
    def validate_funding_policy(self) -> Self:
        if self.funding_mode == "historical":
            if self.funding_rate_column != "funding_rate" or self.default_funding_rate != 0.0:
                raise ValueError("Historical funding requires funding_rate and no default rate")
            if not self.require_funding_event_flag:
                raise ValueError("Historical funding requires explicit funding-event flags")
        return self
