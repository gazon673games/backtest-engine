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
from backtest_engine.intraday.event.engine import (
    IntradayEventBacktestConfig,
    IntradayEventDatasetConfig,
    IntradayEventExecutionConfig,
    IntradayEventInstrumentConfig,
    IntradayEventOutputConfig,
    IntradayEventPortfolioConfig,
    IntradayEventStrategyConfig,
    run_intraday_event_backtest,
)
from backtest_engine.intraday.event.batch.runner import (
    IntradayEventBatchConfig,
    run_intraday_event_batch,
)
from backtest_engine.portfolio.combine import (
    PortfolioCombineConfig,
    combine_intraday_event_portfolio,
)
from backtest_engine.research.pipeline import (
    IntradayEventResearchPipelineConfig,
    run_intraday_event_research_pipeline,
)
from backtest_engine.research.validation.gate import (
    ResearchValidationConfig,
    generate_research_validation_report,
)

runner = CliRunner()


def make_config(
    tmp_path: Path,
    dataset_path: Path,
    plugin_path: Path,
    *,
    run_id: str = "run",
    sizing_mode: str = "signal_units",
    fixed_contracts: float = 1.0,
    initial_capital: float = 1.0,
) -> IntradayEventBacktestConfig:
    return IntradayEventBacktestConfig(
        dataset=IntradayEventDatasetConfig(
            path=dataset_path,
            quality_approval_required=False,
            allow_legacy_bar_timestamps=True,
        ),
        strategy=IntradayEventStrategyConfig(strategy_file=plugin_path),
        instrument=IntradayEventInstrumentConfig(contract_multiplier=1.0, lot_size=1.0),
        portfolio=IntradayEventPortfolioConfig(
            initial_capital=initial_capital,
            sizing_mode=sizing_mode,
            fixed_contracts=fixed_contracts,
        ),
        execution=IntradayEventExecutionConfig(
            timing="signal_close_to_same_close",
            allow_noncausal_same_close=True,
            end_of_data_position_policy="mark",
            from_date=date(2024, 1, 2),
        ),
        output=IntradayEventOutputConfig(root=tmp_path / "out", run_id=run_id),
    )


def write_plugin(tmp_path: Path, source: str, name: str = "strategy.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def write_dataset(
    tmp_path: Path,
    *,
    timestamps: list[datetime] | None = None,
    closes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> Path:
    closes = closes or [100.0, 110.0, 120.0]
    timestamps = timestamps or [
        datetime(2024, 1, 2, 13, 30 + 5 * index) for index in range(len(closes))
    ]
    highs = highs or closes
    lows = lows or closes
    path = tmp_path / "dataset.parquet"
    frame = pl.DataFrame(
        {
            "source": ["fixture"] * len(timestamps),
            "exchange": ["MOEX"] * len(timestamps),
            "engine": ["futures"] * len(timestamps),
            "market": ["forts"] * len(timestamps),
            "board": ["RFUD"] * len(timestamps),
            "symbol": ["CRU6"] * len(timestamps),
            "continuous_symbol": ["CNYRUB_NEAREST_5M"] * len(timestamps),
            "timeframe": ["5m"] * len(timestamps),
            "ts": timestamps,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * len(timestamps),
            "value": [100000.0] * len(timestamps),
            "currency": ["RUB"] * len(timestamps),
            "active_contract": ["CRU6"] * len(timestamps),
            "source_contract_symbol": ["CRU6"] * len(timestamps),
            "dataset_type": ["nearest_contract"] * len(timestamps),
            "adjustment_mode": ["unadjusted"] * len(timestamps),
        }
    )
    frame.write_parquet(path)
    return path


def write_catalog_bundle(tmp_path: Path, dataset_path: Path) -> Path:
    registry_path = tmp_path / "registry.json"
    calendar_path = tmp_path / "calendar.json"
    approval_path = tmp_path / "quality_approval.json"
    catalog_path = tmp_path / "catalog.json"
    registry_path.write_text(
        json.dumps(
            {
                "instruments": [
                    {
                        "symbol": "CR*",
                        "contract_multiplier": 1.0,
                        "lot_size": 1.0,
                        "tick_size": 0.001,
                        "tick_value": 1.0,
                        "initial_margin": 1000.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calendar_path.write_text(
        json.dumps(
            {
                "main_session_start": "10:00",
                "main_session_end": "18:50",
                "holiday_dates": [],
                "short_day_dates": [],
            }
        ),
        encoding="utf-8",
    )
    dataset_sha = file_sha256(dataset_path)
    approval_path.write_text(
        json.dumps(
            {
                "approval_status": "approved",
                "continuous_symbol": "CNYRUB_NEAREST_5M",
                "timeframe": "5m",
                "dataset_sha256": dataset_sha,
                "usable_decision_day_pct": 1.0,
            }
        ),
        encoding="utf-8",
    )
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_version": 1,
                "entries": [
                    {
                        "dataset_id": "fixture:5m:unadjusted",
                        "dataset_path": str(dataset_path),
                        "dataset_sha256": dataset_sha,
                        "quality_approval_path": str(approval_path),
                        "instrument_registry_path": str(registry_path),
                        "calendar_path": str(calendar_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return catalog_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
