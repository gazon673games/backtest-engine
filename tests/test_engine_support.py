# ruff: noqa
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from backtest_engine.daily.batch import BatchConfig, expand_grid, generate_run_id, run_batch
from backtest_engine.cli import app
from backtest_engine.data.dataset import load_dataset, validate_dataset
from backtest_engine.daily.engine import run_backtest
from backtest_engine.core.exceptions import (
    DatasetValidationError,
    QualityGateError,
    StrategyPluginError,
)
from backtest_engine.core.models import (
    BacktestConfig,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
    StrategyConfig,
)
from backtest_engine.strategies.plugins.loader import load_strategy_plugin, strategy_file_sha256
from backtest_engine.reporting.experiment import ReportConfig, generate_experiment_report

runner = CliRunner()


def make_config(
    tmp_path: Path,
    dataset_path: Path,
    manifest_path: Path,
    *,
    strategy: StrategyConfig | None = None,
    execution: ExecutionConfig | None = None,
    run_id: str = "run",
) -> BacktestConfig:
    return BacktestConfig(
        dataset=DatasetConfig(path=dataset_path, quality_manifest=manifest_path),
        strategy=strategy or StrategyConfig(name="buy_and_hold"),
        execution=execution or ExecutionConfig(),
        output=OutputConfig(root=tmp_path / "out", run_id=run_id),
    )


def write_dataset(
    tmp_path: Path,
    *,
    symbol: str = "TEST_CONT",
    adjustment_mode: str = "back_adjusted",
    closes: list[float] | None = None,
    ts: list[datetime] | None = None,
    active_contracts: list[str] | None = None,
) -> Path:
    if closes is None:
        row_count = len(ts) if ts is not None else 5
        closes = [100.0 + index for index in range(row_count)]
    timestamps = ts or [
        datetime(2024, 1, 1) + timedelta(days=index) for index in range(len(closes))
    ]
    active_contracts = active_contracts or ["TESTH4"] * len(closes)
    roll_flags = [
        index > 0 and active_contracts[index - 1] != active_contracts[index]
        for index in range(len(closes))
    ]
    roll_ids: list[int] = []
    roll_id = 0
    for roll_flag in roll_flags:
        roll_id += int(roll_flag)
        roll_ids.append(roll_id)
    dataset_path = tmp_path / f"{symbol}_{adjustment_mode}.parquet"
    df = pl.DataFrame(
        {
            "continuous_symbol": [symbol] * len(closes),
            "timeframe": ["1d"] * len(closes),
            "ts": timestamps,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * len(closes),
            "value": [100000.0] * len(closes),
            "active_contract": active_contracts,
            "source_contract_symbol": active_contracts,
            "roll_id": roll_ids,
            "roll_flag": roll_flags,
            "roll_date": timestamps,
            "adjustment_mode": [adjustment_mode] * len(closes),
            "adjustment_factor": [1.0] * len(closes),
        }
    )
    df.write_parquet(dataset_path)
    return dataset_path


def write_manifest(
    tmp_path: Path,
    *,
    dataset_path: Path,
    status: str,
    symbol: str = "TEST_CONT",
) -> Path:
    manifest_path = tmp_path / "quality_gate_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "dataset": symbol,
                        "path": str(dataset_path),
                        "dataset_sha256": (
                            file_sha256(dataset_path) if dataset_path.exists() else "0" * 64
                        ),
                        "status": status,
                        "reasons": [] if status == "approved" else [f"fixture status {status}"],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_plugin(tmp_path: Path, source: str, filename: str = "strategy.py") -> Path:
    plugin_path = tmp_path / filename
    plugin_path.write_text(source, encoding="utf-8")
    return plugin_path


def write_signal_plugin(tmp_path: Path, signals: list[float]) -> Path:
    return write_plugin(
        tmp_path,
        f"def generate_signals(ctx):\n    return {signals!r}[:len(ctx.series('close'))]\n",
    )


def write_ma_plugin(tmp_path: Path) -> Path:
    return write_plugin(
        tmp_path,
        """
def generate_signals(ctx, params=None):
    params = params or {}
    fast = ctx.sma("close", int(params.get("fast_window", 1)))
    slow = ctx.sma("close", int(params.get("slow_window", 2)))
    return ctx.long_when(fast > slow)
""".lstrip(),
    )


def write_experiment_outputs(tmp_path: Path, *, include_failed: bool = False) -> Path:
    experiment_path = tmp_path / "experiment"
    experiment_path.mkdir()
    rows = [
        experiment_row("CNYRUB_CONT", "cny_slow", 5, 60, 0.3, 0.1),
        experiment_row("CNYRUB_CONT", "cny_fast", 20, 60, 0.7, 0.2),
        experiment_row("USDRUB_CONT", "usd_slow", 10, 120, 0.1, -0.05),
        experiment_row("USDRUB_CONT", "usd_fast", 5, 20, 0.4, 0.15),
    ]
    summary = pl.DataFrame(rows, orient="row")
    summary.write_parquet(experiment_path / "summary.parquet")
    summary.write_csv(experiment_path / "summary.csv")
    (experiment_path / "summary.json").write_text(json.dumps(rows), encoding="utf-8")
    (experiment_path / "experiment_config.json").write_text(
        json.dumps(
            {
                "experiment_id": "fixture_experiment",
                "grid": {"fast_window": [5, 10, 20], "slow_window": [20, 60, 120]},
            }
        ),
        encoding="utf-8",
    )

    failed_rows = []
    if include_failed:
        failed_rows.append(
            {
                "experiment_id": "fixture_experiment",
                "dataset_path": "blocked.parquet",
                "strategy_file": "strategy.py",
                "params": '{"fast_window": 20, "slow_window": 60}',
                "status": "blocked_by_quality_gate",
                "error_type": "QualityGateError",
                "error_message": "Dataset is not approved for backtesting: status=split_required",
            }
        )
    failed = pl.DataFrame(
        failed_rows,
        schema={
            "experiment_id": pl.String,
            "dataset_path": pl.String,
            "strategy_file": pl.String,
            "params": pl.String,
            "status": pl.String,
            "error_type": pl.String,
            "error_message": pl.String,
        },
        orient="row",
    )
    failed.write_parquet(experiment_path / "failed_runs.parquet")
    failed.write_csv(experiment_path / "failed_runs.csv")
    (experiment_path / "failed_runs.json").write_text(json.dumps(failed_rows), encoding="utf-8")
    for row in rows:
        run_path = experiment_path / "runs" / str(row["run_id"])
        run_path.mkdir(parents=True, exist_ok=True)
        write_equity_fixture(run_path / "equity.parquet")
    return experiment_path


def experiment_row(
    symbol: str,
    run_id: str,
    fast_window: int,
    slow_window: int,
    sharpe: float,
    total_return: float,
) -> dict[str, object]:
    return {
        "experiment_id": "fixture_experiment",
        "run_id": run_id,
        "run_path": f"runs/{run_id}",
        "dataset_path": f"{symbol}.parquet",
        "continuous_symbol": symbol,
        "adjustment_mode": "back_adjusted",
        "strategy_type": "plugin",
        "strategy_file": "local_strategy.py",
        "strategy_file_sha256": "abc123",
        "fast_window": fast_window,
        "slow_window": slow_window,
        "commission_bps": 0.0,
        "slippage_bps": 0.0,
        "start_ts": "2024-01-01T00:00:00",
        "end_ts": "2024-01-05T00:00:00",
        "bar_count": 5,
        "total_return": total_return,
        "annualized_return": total_return,
        "annualized_volatility": 0.2,
        "sharpe": sharpe,
        "max_drawdown": -0.1,
        "trade_count": 3,
        "win_rate": 0.5,
        "avg_win": 0.1,
        "avg_loss": -0.05,
        "profit_factor": 2.0,
        "best_trade": 0.15,
        "worst_trade": -0.05,
        "avg_trade_return": 0.03,
        "avg_holding_bars": 4.0,
        "exposure": 0.5,
        "status": "success",
    }


def write_equity_fixture(path: Path) -> None:
    frame = pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 4),
            ],
            "equity": [1.0, 1.1, 1.05, 1.2],
        }
    )
    frame.write_parquet(path)
