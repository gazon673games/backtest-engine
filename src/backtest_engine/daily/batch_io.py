from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import polars as pl

SUMMARY_SCHEMA = {
    "experiment_id": pl.String,
    "run_id": pl.String,
    "run_path": pl.String,
    "dataset_path": pl.String,
    "continuous_symbol": pl.String,
    "adjustment_mode": pl.String,
    "strategy_type": pl.String,
    "strategy_file": pl.String,
    "strategy_file_sha256": pl.String,
    "fast_window": pl.Int64,
    "slow_window": pl.Int64,
    "commission_bps": pl.Float64,
    "slippage_bps": pl.Float64,
    "start_ts": pl.String,
    "end_ts": pl.String,
    "bar_count": pl.Int64,
    "total_return": pl.Float64,
    "annualized_return": pl.Float64,
    "annualized_volatility": pl.Float64,
    "sharpe": pl.Float64,
    "max_drawdown": pl.Float64,
    "trade_count": pl.Int64,
    "win_rate": pl.Float64,
    "avg_win": pl.Float64,
    "avg_loss": pl.Float64,
    "profit_factor": pl.Float64,
    "best_trade": pl.Float64,
    "worst_trade": pl.Float64,
    "avg_trade_return": pl.Float64,
    "avg_holding_bars": pl.Float64,
    "exposure": pl.Float64,
    "status": pl.String,
}

FAILED_SCHEMA = {
    "experiment_id": pl.String,
    "dataset_path": pl.String,
    "strategy_file": pl.String,
    "params": pl.String,
    "status": pl.String,
    "error_type": pl.String,
    "error_message": pl.String,
}


def write_batch_outputs(
    experiment_dir: Path,
    summary_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> None:
    write_table(experiment_dir / "summary", summary_rows, SUMMARY_SCHEMA)
    write_table(experiment_dir / "failed_runs", failed_rows, FAILED_SCHEMA)


def write_table(
    base_path: Path,
    rows: list[dict[str, Any]],
    schema: dict[str, pl.DataType],
) -> None:
    frame = pl.DataFrame(rows, schema=schema, orient="row")
    frame.write_csv(base_path.with_suffix(".csv"))
    frame.write_parquet(base_path.with_suffix(".parquet"))
    write_json(base_path.with_suffix(".json"), rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def safe_path_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("._")
    return safe or "value"
