"""Runs the native portfolio without replacing its execution or ledger."""

import json
from pathlib import Path
from typing import Any

from backtest_engine.portfolio.multi.engine import run_multi_instrument_backtest
from backtest_engine.workbench.history import append_workbench_history, history_path_for
from backtest_engine.workbench.portfolio_api import PortfolioRequest, digest
from backtest_engine.workbench.progress import write_progress


def execute_portfolio(job_id: str, payload: dict[str, Any]) -> None:
    request = PortfolioRequest.model_validate(payload["request"])
    root = Path(payload["output_root"])
    config = request.config
    config.output.root = root
    config.output.run_id = job_id
    record = {
        "run_id": job_id,
        "kind": "portfolio",
        "name": request.name,
        "created_at": payload["created_at"],
        "from_date": str(config.execution.from_date),
        "asset_ids": [leg.asset_id for leg in config.legs],
        "diagnostic": request.diagnostic,
        "fingerprint": payload["fingerprint"],
    }
    write_progress(
        root,
        job_id,
        status="running",
        stage="backtest",
        message="Running native portfolio",
        pct=0.05,
        extra={"job_type": "portfolio"},
    )
    try:
        for path, expected in payload["input_pins"].items():
            if digest(Path(path)) != expected:
                raise ValueError(f"Portfolio input changed after submission: {path}")
        result = run_multi_instrument_backtest(config)
        (result.output_dir / "workbench_portfolio.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        record.update(status="completed", metrics=result.metrics, output_dir=str(result.output_dir))
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        append_workbench_history(history_path_for(root), record)
    write_progress(
        root,
        job_id,
        status="completed",
        stage="done",
        message="Portfolio completed",
        pct=1,
        extra={"job_type": "portfolio", "output_dir": str(result.output_dir)},
    )
