from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.metrics import calculate_metrics
from backtest_engine.core.models import (
    BacktestConfig,
    BacktestResult,
    DatasetInspection,
    DatasetMetadata,
    DatasetQualityResult,
    ExecutionConfig,
)
from backtest_engine.core.run_output import publish_run_output
from backtest_engine.daily.rolls import roll_turnover, validate_roll_returns
from backtest_engine.daily.trade_events import generate_trade_events
from backtest_engine.data.dataset import load_dataset, validate_dataset
from backtest_engine.data.quality import enforce_approved, inspect_quality_gate
from backtest_engine.reporting.trade_attribution import attribute_completed_trades
from backtest_engine.strategies.builtins import generate_signal
from backtest_engine.strategies.lookahead import enforce_vectorized_lookahead
from backtest_engine.strategies.plugins.loader import strategy_file_sha256


def inspect_dataset(dataset_path: Path, quality_manifest: Path) -> DatasetInspection:
    df = load_dataset(dataset_path)
    metadata = validate_dataset(df, dataset_path)
    quality = inspect_quality_gate(dataset_path, quality_manifest, metadata)
    return DatasetInspection(metadata=metadata, quality=quality)


def run_backtest(config: BacktestConfig) -> BacktestResult:
    df = load_dataset(config.dataset.path)
    metadata = validate_dataset(df, config.dataset.path)
    quality = inspect_quality_gate(
        config.dataset.path,
        config.dataset.quality_manifest,
        metadata,
        config.quality_gate,
    )
    enforce_approved(quality)

    equity = build_equity_curve(df, metadata, config)
    trade_events = generate_trade_events(equity)
    trades = attribute_completed_trades(equity)
    metrics = calculate_metrics(equity, trades, config.execution)
    output_dir = save_outputs(
        config,
        quality,
        equity,
        trades,
        trade_events,
        metrics.model_dump(mode="json"),
    )

    return BacktestResult(output_dir=output_dir, metrics=metrics, dataset_quality=quality)


def build_equity_curve(
    df: pl.DataFrame,
    metadata: DatasetMetadata,
    config: BacktestConfig,
) -> pl.DataFrame:
    closes = [float(value) for value in df.get_column("close").to_list()]
    _validate_return_prices(closes)
    active_contracts = [str(value) for value in df.get_column("active_contract").to_list()]

    enforce_vectorized_lookahead(df, config.strategy)
    signal = generate_signal(df, config.strategy)
    position = signal[:-1]
    position.insert(0, 0.0)
    validate_roll_returns(metadata, active_contracts, position)
    return_1d = _close_to_close_returns(closes)
    strategy_return = [
        position_value * return_value
        for position_value, return_value in zip(position, return_1d, strict=True)
    ]
    position_turnover = _turnover(position)
    end_of_data_turnover = [0.0] * len(position)
    if position:
        end_of_data_turnover[-1] = abs(position[-1])
        position_turnover[-1] += end_of_data_turnover[-1]
    roll_turnover_values = roll_turnover(position, active_contracts)
    turnover = [
        position_value + roll_value
        for position_value, roll_value in zip(
            position_turnover,
            roll_turnover_values,
            strict=True,
        )
    ]
    position_cost = _cost(position_turnover, config.execution)
    roll_cost = _cost(roll_turnover_values, config.execution)
    cost = [
        position_value + roll_value
        for position_value, roll_value in zip(position_cost, roll_cost, strict=True)
    ]
    strategy_return_after_cost = [
        gross - cost_value for gross, cost_value in zip(strategy_return, cost, strict=True)
    ]
    equity = _equity(strategy_return_after_cost)

    return pl.DataFrame(
        {
            "ts": df.get_column("ts"),
            "continuous_symbol": [metadata.continuous_symbol] * df.height,
            "adjustment_mode": [metadata.adjustment_mode] * df.height,
            "close": closes,
            "signal": signal,
            "position": position,
            "return_1d": return_1d,
            "strategy_return": strategy_return,
            "position_turnover": position_turnover,
            "end_of_data_turnover": end_of_data_turnover,
            "roll_turnover": roll_turnover_values,
            "turnover": turnover,
            "position_cost": position_cost,
            "roll_cost": roll_cost,
            "cost": cost,
            "strategy_return_after_cost": strategy_return_after_cost,
            "equity": equity,
            "active_contract": df.get_column("active_contract"),
            "source_contract_symbol": df.get_column("source_contract_symbol"),
            "roll_flag": df.get_column("roll_flag"),
            "roll_date": df.get_column("roll_date"),
        }
    )


def save_outputs(
    config: BacktestConfig,
    quality: DatasetQualityResult,
    equity: pl.DataFrame,
    trades: pl.DataFrame,
    trade_events: pl.DataFrame,
    metrics: dict[str, Any],
) -> Path:
    def write(output_dir: Path, published_dir: Path) -> None:
        del published_dir
        _write_json(output_dir / "config.json", _config_output_payload(config))
        _write_json(output_dir / "dataset_quality.json", quality.model_dump(mode="json"))
        equity.write_parquet(output_dir / "equity.parquet")
        trades.write_parquet(output_dir / "trades.parquet")
        trade_events.write_parquet(output_dir / "trade_events.parquet")
        if config.output.write_csv:
            equity.write_csv(output_dir / "equity.csv")
            trades.write_csv(output_dir / "trades.csv")
            trade_events.write_csv(output_dir / "trade_events.csv")
        _write_json(output_dir / "metrics.json", metrics)

    return publish_run_output(config.output.root, config.output.run_id, write, reuse_identical=True)


def _validate_return_prices(closes: list[float]) -> None:
    for close in closes:
        if close == 0:
            raise DatasetValidationError("close must not contain zero values")


def _close_to_close_returns(closes: list[float]) -> list[float]:
    returns = [0.0]
    for previous_close, close in zip(closes, closes[1:], strict=False):
        returns.append(close / previous_close - 1.0)
    return returns


def _turnover(position: list[float]) -> list[float]:
    turnover = []
    previous = 0.0
    for current in position:
        turnover.append(abs(current - previous))
        previous = current
    return turnover


def _cost(turnover: list[float], execution: ExecutionConfig) -> list[float]:
    cost_rate = (execution.commission_bps + execution.slippage_bps) / 10_000.0
    return [turnover_value * cost_rate for turnover_value in turnover]


def _equity(strategy_return_after_cost: list[float]) -> list[float]:
    equity = []
    current = 1.0
    for return_value in strategy_return_after_cost:
        current *= 1.0 + return_value
        equity.append(current)
    return equity


def _config_output_payload(config: BacktestConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    payload["execution_timing"] = config.execution.timing
    payload["commission_bps"] = config.execution.commission_bps
    payload["slippage_bps"] = config.execution.slippage_bps

    if config.strategy.name == "plugin":
        strategy_file = config.strategy.strategy_file
        payload["strategy_type"] = "plugin"
        payload["strategy_file"] = str(strategy_file) if strategy_file is not None else None
        payload["strategy_file_sha256"] = (
            strategy_file_sha256(strategy_file) if strategy_file is not None else None
        )
        payload["strategy_params"] = dict(config.strategy.strategy_params)
    else:
        payload["strategy_type"] = "builtin"

    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
