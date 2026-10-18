from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.run_output import publish_run_output
from backtest_engine.intraday.config import IntradayBacktestConfig
from backtest_engine.strategies.plugins.loader import strategy_file_sha256


def save_intraday_outputs(
    config: IntradayBacktestConfig,
    dataset_validation: dict[str, Any],
    decision_points: pl.DataFrame,
    equity: pl.DataFrame,
    trades: pl.DataFrame,
    skipped_days: pl.DataFrame,
    metrics: dict[str, Any],
    quality_approval: dict[str, Any],
) -> Path:
    def write(output_dir: Path, published_dir: Path) -> None:
        del published_dir
        write_json(output_dir / "config.json", config_output_payload(config, dataset_validation))
        write_json(output_dir / "dataset_validation.json", dataset_validation)
        write_json(output_dir / "quality_approval.json", quality_approval)
        decision_points.write_parquet(output_dir / "decision_points.parquet")
        equity.write_parquet(output_dir / "equity.parquet")
        trades.write_parquet(output_dir / "trades.parquet")
        skipped_days.write_parquet(output_dir / "skipped_decision_days.parquet")
        if config.output.write_csv:
            decision_points.write_csv(output_dir / "decision_points.csv")
            equity.write_csv(output_dir / "equity.csv")
            trades.write_csv(output_dir / "trades.csv")
            skipped_days.write_csv(output_dir / "skipped_decision_days.csv")
        write_json(output_dir / "metrics.json", metrics)

    return publish_run_output(config.output.root, config.output.run_id, write)


def config_output_payload(
    config: IntradayBacktestConfig,
    dataset_validation: dict[str, Any],
) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    payload["strategy_type"] = "intraday_plugin"
    payload["strategy_file"] = str(config.strategy.strategy_file)
    payload["strategy_file_sha256"] = strategy_file_sha256(config.strategy.strategy_file)
    payload["strategy_params"] = dict(config.strategy.strategy_params)
    payload["execution_timing"] = config.execution.timing
    payload["quality_gate_required"] = config.dataset.quality_approval_required
    payload["timeframe"] = dataset_validation["timeframe"]
    payload["bar_minutes"] = dataset_validation["bar_minutes"]
    payload["dataset_type"] = dataset_validation["dataset_type"]
    payload["adjustment_mode"] = dataset_validation["adjustment_mode"]
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
