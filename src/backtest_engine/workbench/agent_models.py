"""Small agent-facing inputs; dataset and strategy identities resolve through catalogs."""

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    strategy_id: str
    from_date: date
    initial_capital: float = Field(gt=0)
    base_currency: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    execution_overrides: dict[str, Any] = Field(default_factory=dict)
