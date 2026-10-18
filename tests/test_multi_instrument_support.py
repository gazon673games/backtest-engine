# ruff: noqa
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from backtest_engine.cli import app
from backtest_engine.data.lifecycle import DataLifecycleAuditConfig, audit_data_lifecycle
from backtest_engine.portfolio.multi.engine import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentExecutionConfig,
    MultiInstrumentLegConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
    run_multi_instrument_backtest,
)

runner = CliRunner()


def write_strategy(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "strategy.py"
    path.write_text(source, encoding="utf-8")
    return path


def write_intraday_dataset(
    tmp_path: Path,
    filename: str,
    asset_id: str,
    closes: list[float],
) -> Path:
    timestamps = [
        datetime(2024, 1, 2, 10, 0),
        datetime(2024, 1, 2, 10, 5),
        datetime(2024, 1, 2, 10, 10),
    ][: len(closes)]
    path = tmp_path / filename
    pl.DataFrame(
        {
            "source": ["fixture"] * len(closes),
            "exchange": ["MOEX"] * len(closes),
            "engine": ["futures"] * len(closes),
            "market": ["forts"] * len(closes),
            "board": ["RFUD"] * len(closes),
            "symbol": [asset_id] * len(closes),
            "continuous_symbol": [f"{asset_id}_NEAREST_5M"] * len(closes),
            "timeframe": ["5m"] * len(closes),
            "ts": timestamps,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * len(closes),
            "value": [100000.0] * len(closes),
            "currency": ["RUB"] * len(closes),
            "active_contract": [asset_id] * len(closes),
            "source_contract_symbol": [asset_id] * len(closes),
            "dataset_type": ["nearest_contract"] * len(closes),
            "adjustment_mode": ["unadjusted"] * len(closes),
        }
    ).write_parquet(path)
    return path


def write_ohlc_dataset(
    tmp_path: Path,
    filename: str,
    asset_id: str,
    *,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> Path:
    timestamps = [
        datetime(2024, 1, 2, 10, 0),
        datetime(2024, 1, 2, 10, 5),
        datetime(2024, 1, 2, 10, 10),
    ][: len(closes)]
    path = tmp_path / filename
    pl.DataFrame(
        {
            "source": ["fixture"] * len(closes),
            "exchange": ["MOEX"] * len(closes),
            "engine": ["futures"] * len(closes),
            "market": ["forts"] * len(closes),
            "board": ["RFUD"] * len(closes),
            "symbol": [asset_id] * len(closes),
            "continuous_symbol": [f"{asset_id}_NEAREST_5M"] * len(closes),
            "timeframe": ["5m"] * len(closes),
            "ts": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * len(closes),
            "value": [100000.0] * len(closes),
            "currency": ["RUB"] * len(closes),
            "active_contract": [asset_id] * len(closes),
            "source_contract_symbol": [asset_id] * len(closes),
            "dataset_type": ["nearest_contract"] * len(closes),
            "adjustment_mode": ["unadjusted"] * len(closes),
        }
    ).write_parquet(path)
    return path


def write_spot_dataset(
    tmp_path: Path,
    filename: str,
    asset_id: str,
    closes: list[float],
) -> Path:
    timestamps = [
        datetime(2024, 1, 2, 10, 0),
        datetime(2024, 1, 2, 10, 5),
        datetime(2024, 1, 2, 10, 10),
    ][: len(closes)]
    path = tmp_path / filename
    pl.DataFrame(
        {
            "source": ["fixture"] * len(closes),
            "exchange": ["CRYPTO"] * len(closes),
            "engine": ["spot"] * len(closes),
            "market": ["crypto"] * len(closes),
            "board": ["SPOT"] * len(closes),
            "symbol": [asset_id] * len(closes),
            "timeframe": ["5m"] * len(closes),
            "ts": timestamps,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * len(closes),
            "value": [100000.0] * len(closes),
            "currency": ["USDT"] * len(closes),
            "dataset_type": ["spot"] * len(closes),
            "base_currency": ["BTC"] * len(closes),
            "quote_currency": ["USDT"] * len(closes),
        }
    ).write_parquet(path)
    return path


def write_catalog(
    tmp_path: Path,
    dataset: Path,
    *,
    broken_registry: bool = False,
    registry_payload: dict[str, object] | None = None,
) -> Path:
    approval = tmp_path / "quality_approval.json"
    registry = tmp_path / "instrument_registry.json"
    calendar = tmp_path / "calendar.json"
    catalog = tmp_path / "catalog.json"
    frame = pl.read_parquet(dataset)
    approval.write_text(
        json.dumps(
            {
                "approval_status": "approved",
                "continuous_symbol": str(frame.get_column("continuous_symbol")[0]),
                "timeframe": str(frame.get_column("timeframe")[0]),
                "dataset_sha256": file_sha256(dataset),
                "usable_decision_day_pct": 1.0,
            }
        ),
        encoding="utf-8",
    )
    if registry_payload is None:
        registry_payload = (
            {"instruments": [{"symbol": "*"}]}
            if broken_registry
            else {
                "instruments": [
                    {
                        "symbol": "*",
                        "contract_multiplier": 1.0,
                        "lot_size": 1.0,
                        "tick_size": 0.001,
                    }
                ]
            }
        )
    registry.write_text(json.dumps(registry_payload), encoding="utf-8")
    calendar.write_text(
        json.dumps(
            {
                "main_session_start": "10:00",
                "main_session_end": "18:50",
                "holiday_dates": [],
            }
        ),
        encoding="utf-8",
    )
    catalog.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "dataset_id": "A",
                        "dataset_path": str(dataset),
                        "dataset_sha256": file_sha256(dataset),
                        "continuous_symbol": "A_NEAREST_5M",
                        "timeframe": "5m",
                        "adjustment_mode": "unadjusted",
                        "row_count": frame.height,
                        "quality_approval_path": str(approval),
                        "instrument_registry_path": str(registry),
                        "calendar_path": str(calendar),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return catalog


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
