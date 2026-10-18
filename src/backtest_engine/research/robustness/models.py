"""Strategy-neutral experiment inputs; parameter names are supplied by the author."""

from datetime import date
from math import prod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Case(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    group: str = "control"
    params: dict[str, Any] = Field(default_factory=dict)
    request_overrides: dict[str, Any] = Field(default_factory=dict)
    existing_run: Path | None = None


class Grid(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,59}$")
    group: str = "neighbors"
    params: dict[str, list[str | int | float | bool]]

    @model_validator(mode="after")
    def nonempty(self):
        if not self.params or any(not values for values in self.params.values()):
            raise ValueError("grid params and every value list must be nonempty")
        return self


class Protocol(StrictModel):
    version: int = Field(default=1, ge=1, le=1)
    name: str
    base_request: Path | None = None
    output_root: Path
    split_date: date
    baseline: str = "baseline"
    cases: list[Case] = Field(default_factory=lambda: [Case(name="baseline", group="baseline")])
    grids: list[Grid] = Field(default_factory=list)
    family_groups: list[str] = Field(default_factory=lambda: ["neighbors"])
    bootstrap_blocks_days: list[int] = Field(default_factory=lambda: [5, 10, 20])
    bootstrap_iterations: int = Field(default=1000, ge=0, le=5000)
    seed: int = 20261008
    selection_bias_iterations: int = Field(default=1000, ge=0, le=5000)
    workers: int = Field(default=1, ge=1, le=4)
    max_runs: int = Field(default=500, ge=1, le=500)
    fresh_selected_group: str | None = None
    history_previously_inspected: bool = True
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def positive_blocks(self):
        count = len(self.cases) + sum(prod(len(v) for v in g.params.values()) for g in self.grids)
        if count + bool(self.fresh_selected_group) > self.max_runs:
            raise ValueError(f"protocol exceeds max_runs={self.max_runs}")
        if any(block < 1 for block in self.bootstrap_blocks_days):
            raise ValueError("bootstrap_blocks_days must contain positive integers")
        if len(set(self.bootstrap_blocks_days)) != len(self.bootstrap_blocks_days):
            raise ValueError("bootstrap_blocks_days must be unique")
        if len(set(self.family_groups)) != len(self.family_groups):
            raise ValueError("family_groups must be unique")
        return self
