from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backtest_engine.research.external_validation import (
    ExternalValidationConfig,
    validate_against_statement,
)
from backtest_engine.workbench.runner import safe_run_id


class WorkbenchValidationRequest(BaseModel):
    run_id: str
    statement_path: Path
    tolerance: float = Field(default=0.01, ge=0.0)


def validate_workbench_run(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    request = WorkbenchValidationRequest.model_validate(payload)
    run_id = safe_run_id(request.run_id)
    output_dir = output_root / run_id
    events_path = output_dir / "events.parquet"
    result = validate_against_statement(
        ExternalValidationConfig(
            backtest_events_path=events_path,
            statement_path=request.statement_path,
            output_path=output_dir / "external_validation.json",
            tolerance=request.tolerance,
        )
    )
    report = json.loads(result.output_path.read_text(encoding="utf-8"))
    return {"run_id": run_id, "output_path": str(result.output_path), **report}
