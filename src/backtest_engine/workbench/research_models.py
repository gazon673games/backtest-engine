from __future__ import annotations

import math
from datetime import date

from pydantic import BaseModel, Field, model_validator

from backtest_engine.workbench.models import WorkbenchRunRequest


class WorkbenchSweepRequest(BaseModel):
    input_hashes: dict[str, str] = Field(default_factory=dict)
    base_request: WorkbenchRunRequest
    param_grid: dict[str, list[str | int | float | bool]]
    split_date: date | None = None
    run_id: str | None = None
    max_runs: int = Field(default=100, ge=1, le=500)
    bootstrap_iterations: int = Field(default=200, ge=0, le=5000)
    cost_stress_bps: list[float] = Field(default_factory=lambda: [1.0, 2.0, 5.0])

    @model_validator(mode="after")
    def validate_grid(self) -> WorkbenchSweepRequest:
        if not self.param_grid:
            raise ValueError("param_grid must not be empty")
        empty = [name for name, values in self.param_grid.items() if not values]
        if empty:
            raise ValueError("param_grid values must not be empty: " + ", ".join(empty))
        run_count = math.prod(len(values) for values in self.param_grid.values())
        if run_count > self.max_runs:
            raise ValueError(f"parameter grid has {run_count} runs; max_runs={self.max_runs}")
        return self

    @property
    def run_count(self) -> int:
        return math.prod(len(values) for values in self.param_grid.values())
