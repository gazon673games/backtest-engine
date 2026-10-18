from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.provenance import runtime_provenance
from backtest_engine.intraday.event.batch.config import (
    IntradayEventBatchConfig,
    IntradayEventBatchResult,
)
from backtest_engine.intraday.event.batch.io import write_json, write_table
from backtest_engine.intraday.event.batch.parsing import (
    decision_time_text,
    file_sha256,
    optional_string,
)
from backtest_engine.intraday.event.engine import IntradayEventExecutionConfig
from backtest_engine.research.experiment_identity import canonical_fingerprint
from backtest_engine.strategies.plugins.loader import strategy_file_sha256

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
    "catalog_enforcement": pl.String,
    "quality_approval_status": pl.String,
    "schedule": pl.String,
    "execution_timing": pl.String,
    "decision_time": pl.String,
    "missing_decision_policy": pl.String,
    "from_date": pl.String,
    "till_date": pl.String,
    "window_minutes": pl.String,
    "window_bars": pl.String,
    "commission_bps": pl.Float64,
    "slippage_bps": pl.Float64,
    "slippage_ticks": pl.Float64,
    "bid_ask_spread_bps": pl.Float64,
    "bid_ask_spread_ticks": pl.Float64,
    "volume_participation_limit": pl.Float64,
    "volume_to_contracts_multiplier": pl.Float64,
    "partial_fill_policy": pl.String,
    "intrabar_ambiguity_policy": pl.String,
    "roll_cost_bps": pl.Float64,
    "roll_slippage_ticks": pl.Float64,
    "roll_pnl_policy": pl.String,
    "bar_count": pl.Int64,
    "event_count": pl.Int64,
    "strategy_call_count": pl.Int64,
    "order_count": pl.Int64,
    "filled_order_count": pl.Int64,
    "unfilled_order_count": pl.Int64,
    "partial_fill_count": pl.Int64,
    "volume_limited_count": pl.Int64,
    "volume_rejected_count": pl.Int64,
    "missing_decision_count": pl.Int64,
    "missing_decision_close_count": pl.Int64,
    "total_return": pl.Float64,
    "annualized_return": pl.Float64,
    "annualized_volatility": pl.Float64,
    "sharpe": pl.Float64,
    "max_drawdown": pl.Float64,
    "sortino": pl.Float64,
    "calmar": pl.Float64,
    "value_at_risk_95": pl.Float64,
    "expected_shortfall_95": pl.Float64,
    "return_skewness": pl.Float64,
    "return_excess_kurtosis": pl.Float64,
    "max_drawdown_duration_periods": pl.Int64,
    "sharpe_ci_95_low": pl.Float64,
    "sharpe_ci_95_high": pl.Float64,
    "trade_count": pl.Int64,
    "long_trade_count": pl.Int64,
    "short_trade_count": pl.Int64,
    "win_rate": pl.Float64,
    "avg_win": pl.Float64,
    "avg_loss": pl.Float64,
    "profit_factor": pl.Float64,
    "best_trade": pl.Float64,
    "worst_trade": pl.Float64,
    "avg_trade_return": pl.Float64,
    "avg_holding_bars": pl.Float64,
    "exposure": pl.Float64,
    "long_exposure": pl.Float64,
    "short_exposure": pl.Float64,
    "turnover": pl.Float64,
    "total_cost": pl.Float64,
    "commission_cost_cash": pl.Float64,
    "slippage_cost_cash": pl.Float64,
    "spread_cost_cash": pl.Float64,
    "roll_cost_cash": pl.Float64,
    "trading_fee_cash": pl.Float64,
    "funding_cost_cash": pl.Float64,
    "liquidation_fee_cash": pl.Float64,
    "liquidation_count": pl.Int64,
    "max_maintenance_margin": pl.Float64,
    "final_equity_cash": pl.Float64,
    "net_pnl": pl.Float64,
    "turnover_notional": pl.Float64,
    "max_position_notional": pl.Float64,
    "status": pl.String,
    "cache_status": pl.String,
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
    config: IntradayEventBatchConfig,
    run_id: str,
    run_path: Path,
    dataset_path: Path,
    dataset_validation: dict[str, Any],
    quality_approval: dict[str, Any],
    execution: IntradayEventExecutionConfig,
    plugin_params: dict[str, Any],
    metrics: dict[str, Any],
    cache_status: str = "miss",
) -> dict[str, Any]:
    row = {
        "experiment_id": config.experiment_id,
        "run_id": run_id,
        "run_path": str(run_path),
        "dataset_path": str(dataset_path),
        "continuous_symbol": dataset_validation["continuous_symbol"],
        "timeframe": dataset_validation["timeframe"],
        "dataset_type": dataset_validation["dataset_type"],
        "adjustment_mode": dataset_validation["adjustment_mode"],
        "strategy_file": str(config.strategy_file),
        "strategy_file_sha256": strategy_file_sha256(config.strategy_file),
        "catalog_enforcement": config.catalog_enforcement,
        "quality_approval_status": quality_approval.get("status"),
        "schedule": execution.schedule,
        "execution_timing": execution.timing,
        "decision_time": decision_time_text(execution.decision_times),
        "missing_decision_policy": execution.missing_decision_policy,
        "from_date": execution.from_date.isoformat(),
        "till_date": None,
        "window_minutes": optional_string(plugin_params.get("window_minutes")),
        "window_bars": optional_string(plugin_params.get("window_bars")),
        **execution_columns(execution),
        **metrics_columns(metrics),
        "status": "success",
        "cache_status": cache_status,
    }
    for key, value in plugin_params.items():
        row.setdefault(key, value)
    return row


def execution_columns(execution: IntradayEventExecutionConfig) -> dict[str, Any]:
    return {
        "commission_bps": execution.commission_bps,
        "slippage_bps": execution.slippage_bps,
        "slippage_ticks": execution.slippage_ticks,
        "bid_ask_spread_bps": execution.bid_ask_spread_bps,
        "bid_ask_spread_ticks": execution.bid_ask_spread_ticks,
        "volume_participation_limit": execution.volume_participation_limit,
        "volume_to_contracts_multiplier": execution.volume_to_contracts_multiplier,
        "partial_fill_policy": execution.partial_fill_policy,
        "intrabar_ambiguity_policy": execution.intrabar_ambiguity_policy,
        "roll_cost_bps": execution.roll_cost_bps,
        "roll_slippage_ticks": execution.roll_slippage_ticks,
        "roll_pnl_policy": execution.roll_pnl_policy,
    }


def metrics_columns(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in SUMMARY_BASE_SCHEMA if key in metrics}


def failed_row(
    config: IntradayEventBatchConfig,
    dataset_path: Path,
    params: dict[str, Any],
    exc: Exception,
) -> dict[str, str]:
    return {
        "experiment_id": config.experiment_id,
        "dataset_path": str(dataset_path),
        "strategy_file": str(config.strategy_file),
        "params": json.dumps(params, sort_keys=True),
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def batch_result(
    experiment_dir: Path,
    summary_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    *,
    skipped_duplicate_count: int = 0,
) -> IntradayEventBatchResult:
    return IntradayEventBatchResult(
        output_dir=experiment_dir,
        successful_run_count=len(summary_rows),
        failed_run_count=len(failed_rows),
        resumed_run_count=sum(row.get("cache_status") == "hit" for row in summary_rows),
        skipped_duplicate_count=skipped_duplicate_count,
        summary_rows=summary_rows,
        failed_rows=failed_rows,
    )


def write_batch_outputs(
    experiment_dir: Path,
    summary_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> None:
    write_table(experiment_dir / "summary", summary_rows, SUMMARY_BASE_SCHEMA)
    write_table(experiment_dir / "failed_runs", failed_rows, FAILED_SCHEMA)


def write_experiment_manifest(
    experiment_dir: Path,
    config: IntradayEventBatchConfig,
    summary_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    *,
    skipped_duplicate_count: int = 0,
) -> None:
    run_manifests = []
    for row in summary_rows:
        manifest_path = Path(str(row["run_path"])) / "experiment_manifest.json"
        run_manifests.append(
            {
                "run_id": row["run_id"],
                "run_path": row["run_path"],
                "manifest_path": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path) if manifest_path.exists() else None,
            }
        )
    write_json(
        experiment_dir / "experiment_manifest.json",
        {
            "manifest_version": 2,
            "runtime": runtime_provenance(),
            "experiment_fingerprint": canonical_fingerprint(
                config.model_dump(mode="json", exclude={"output_root", "max_workers"})
            ),
            "experiment_id": config.experiment_id,
            "output_dir": str(experiment_dir),
            "config": config.model_dump(mode="json"),
            "run_count": len(summary_rows),
            "failed_count": len(failed_rows),
            "resumed_count": sum(row.get("cache_status") == "hit" for row in summary_rows),
            "skipped_duplicate_count": skipped_duplicate_count,
            "run_manifests": run_manifests,
            "failed_runs": failed_rows,
            "outputs": {
                name: {
                    "path": str(experiment_dir / name),
                    "sha256": file_sha256(experiment_dir / name),
                }
                for name in ("summary.csv", "summary.parquet", "failed_runs.csv")
                if (experiment_dir / name).exists()
            },
        },
    )
