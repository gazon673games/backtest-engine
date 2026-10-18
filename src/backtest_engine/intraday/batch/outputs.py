from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.intraday.batch.config import IntradayBatchConfig
from backtest_engine.intraday.batch.parsing import optional_string, strategy_sha256
from backtest_engine.intraday.engine import IntradayExecutionConfig

SUMMARY_BASE_SCHEMA = {
    "experiment_id": pl.String,
    "run_id": pl.String,
    "run_path": pl.String,
    "dataset_path": pl.String,
    "continuous_symbol": pl.String,
    "timeframe": pl.String,
    "dataset_type": pl.String,
    "adjustment_mode": pl.String,
    "strategy_file": pl.String,
    "strategy_file_sha256": pl.String,
    "from_date": pl.String,
    "till_date": pl.String,
    "decision_time": pl.String,
    "decision_tolerance_minutes": pl.Int64,
    "minimum_bars_since_session_start": pl.Int64,
    "window_bars": pl.String,
    "window_minutes": pl.String,
    "commission_bps": pl.Float64,
    "slippage_bps": pl.Float64,
    "decision_count": pl.Int64,
    "skipped_decision_day_count": pl.Int64,
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


def summary_row(
    *,
    config: IntradayBatchConfig,
    run_id: str,
    run_path: Path,
    dataset_validation: dict[str, Any],
    execution: IntradayExecutionConfig,
    plugin_params: dict[str, Any],
    metrics: dict[str, Any],
    trade_metrics: dict[str, float | None],
) -> dict[str, Any]:
    row = {
        "experiment_id": config.experiment_id,
        "run_id": run_id,
        "run_path": str(run_path),
        "dataset_path": str(config.dataset_path),
        "continuous_symbol": dataset_validation["continuous_symbol"],
        "timeframe": dataset_validation["timeframe"],
        "dataset_type": dataset_validation["dataset_type"],
        "adjustment_mode": dataset_validation["adjustment_mode"],
        "strategy_file": str(config.strategy_file),
        "strategy_file_sha256": strategy_sha256(config.strategy_file),
        "from_date": execution.from_date.isoformat(),
        "till_date": None,
        "decision_time": execution.decision_time.isoformat(timespec="minutes"),
        "decision_tolerance_minutes": execution.decision_tolerance_minutes,
        "minimum_bars_since_session_start": execution.minimum_bars_since_session_start,
        "window_bars": optional_string(plugin_params.get("window_bars")),
        "window_minutes": optional_string(plugin_params.get("window_minutes")),
        "commission_bps": execution.commission_bps,
        "slippage_bps": execution.slippage_bps,
        **metric_columns(metrics),
        "best_trade": trade_metrics["best_trade"],
        "worst_trade": trade_metrics["worst_trade"],
        "avg_trade_return": trade_metrics["avg_trade_return"],
        "avg_holding_bars": trade_metrics["avg_holding_bars"],
        "exposure": metrics.get("exposure"),
        "status": "success",
    }
    for key, value in plugin_params.items():
        row.setdefault(key, value)
    return row


def metric_columns(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_count": metrics.get("decision_count"),
        "skipped_decision_day_count": metrics.get("skipped_decision_day_count"),
        "total_return": metrics.get("total_return"),
        "annualized_return": metrics.get("annualized_return"),
        "annualized_volatility": metrics.get("annualized_volatility"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "trade_count": metrics.get("trade_count"),
        "win_rate": metrics.get("win_rate"),
        "avg_win": metrics.get("avg_win"),
        "avg_loss": metrics.get("avg_loss"),
        "profit_factor": metrics.get("profit_factor"),
    }


def failed_row(
    config: IntradayBatchConfig,
    params: dict[str, Any],
    exc: Exception,
) -> dict[str, str]:
    return {
        "experiment_id": config.experiment_id,
        "dataset_path": str(config.dataset_path),
        "strategy_file": str(config.strategy_file),
        "params": json.dumps(params, sort_keys=True),
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def trade_summary_metrics(trades: pl.DataFrame) -> dict[str, float | None]:
    if trades.height == 0:
        return {
            "best_trade": None,
            "worst_trade": None,
            "avg_trade_return": None,
            "avg_holding_bars": None,
        }
    net_returns = [float(value) for value in trades.get_column("net_return").to_list()]
    holding = [float(value) for value in trades.get_column("holding_intervals").to_list()]
    return {
        "best_trade": max(net_returns),
        "worst_trade": min(net_returns),
        "avg_trade_return": sum(net_returns) / len(net_returns),
        "avg_holding_bars": sum(holding) / len(holding),
    }


def write_batch_outputs(
    experiment_dir: Path,
    summary_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> None:
    write_table(experiment_dir / "summary", summary_rows, SUMMARY_BASE_SCHEMA)
    write_table(experiment_dir / "failed_runs", failed_rows, FAILED_SCHEMA)


def write_table(
    base_path: Path,
    rows: list[dict[str, Any]],
    base_schema: dict[str, pl.DataType],
) -> None:
    if rows:
        schema = {
            **{key: dtype for key, dtype in base_schema.items() if all(key in row for row in rows)},
            **{key: pl.String for row in rows for key in row if key not in base_schema},
        }
        normalized_rows = [stringify_dynamic_columns(row, base_schema) for row in rows]
        frame = pl.DataFrame(normalized_rows, schema=schema, orient="row")
    else:
        frame = pl.DataFrame(rows, schema=base_schema, orient="row")
    frame.write_csv(base_path.with_suffix(".csv"))
    frame.write_parquet(base_path.with_suffix(".parquet"))
    write_json(base_path.with_suffix(".json"), rows)


def stringify_dynamic_columns(
    row: dict[str, Any],
    base_schema: dict[str, pl.DataType],
) -> dict[str, Any]:
    return {
        key: (optional_string(value) if key not in base_schema else value)
        for key, value in row.items()
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
