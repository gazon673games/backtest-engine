from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class IntradayEventBacktestResult(BaseModel):
    output_dir: Path
    metrics: dict[str, Any]
    dataset_validation: dict[str, Any]
    quality_approval: dict[str, Any] = Field(default_factory=dict)
    catalog_approval: dict[str, Any] = Field(default_factory=dict)
