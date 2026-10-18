# ruff: noqa
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl


def write_intraday_dataset(
    tmp_path: Path,
    *,
    timestamps: list[datetime] | None = None,
    closes: list[float] | None = None,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
    active_contracts: list[str] | None = None,
    timeframe: str = "5m",
) -> Path:
    timestamps = timestamps or [
        datetime(2024, 1, 2, 13, 30),
        datetime(2024, 1, 2, 13, 35),
    ]
    closes = closes or [100.0 + index * 10.0 for index in range(len(timestamps))]
    opens = opens or closes
    highs = highs or [max(open_price, close) for open_price, close in zip(opens, closes)]
    lows = lows or [min(open_price, close) for open_price, close in zip(opens, closes)]
    volumes = volumes if volumes is not None else [1000.0] * len(timestamps)
    dataset_path = tmp_path / "intraday.parquet"
    active_contracts = active_contracts or ["CRH4"] * len(timestamps)
    df = pl.DataFrame(
        {
            "source": ["fixture"] * len(timestamps),
            "exchange": ["MOEX"] * len(timestamps),
            "engine": ["futures"] * len(timestamps),
            "market": ["forts"] * len(timestamps),
            "board": ["RFUD"] * len(timestamps),
            "symbol": active_contracts,
            "continuous_symbol": [f"CNYRUB_NEAREST_{timeframe.upper()}"] * len(timestamps),
            "timeframe": [timeframe] * len(timestamps),
            "ts": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "value": [100000.0] * len(timestamps),
            "currency": ["RUB"] * len(timestamps),
            "active_contract": active_contracts,
            "source_contract_symbol": active_contracts,
            "dataset_type": ["nearest_contract"] * len(timestamps),
            "adjustment_mode": ["unadjusted"] * len(timestamps),
        }
    )
    df.write_parquet(dataset_path)
    return dataset_path


def missing_decision_timestamps() -> list[datetime]:
    return [
        datetime(2024, 1, 2, 13, 30),
        datetime(2024, 1, 2, 13, 35),
        datetime(2024, 1, 3, 13, 20),
        datetime(2024, 1, 3, 13, 35),
        datetime(2024, 1, 4, 13, 30),
        datetime(2024, 1, 4, 13, 35),
    ]


def session_timestamps(start: datetime, *, bars: int) -> list[datetime]:
    return [start + timedelta(minutes=5 * index) for index in range(bars)]


def write_quality_approval(
    tmp_path: Path,
    *,
    recommended_start_date: str = "2024-01-02",
    usable_decision_day_pct: float = 1.0,
) -> Path:
    dataset_path = tmp_path / "intraday.parquet"
    approval_path = tmp_path / "gap_audit_summary.json"
    approval_path.write_text(
        json.dumps(
            {
                "approval_status": "approved",
                "continuous_symbol": "CNYRUB_NEAREST_5M",
                "timeframe": "5m",
                "dataset_sha256": file_sha256(dataset_path),
                "decision_time": "13:30",
                "usable_decision_day_pct": usable_decision_day_pct,
                "recommended_start_date": recommended_start_date,
            }
        ),
        encoding="utf-8",
    )
    return approval_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_intraday_plugin(tmp_path: Path, source: str) -> Path:
    plugin_path = tmp_path / "intraday_event_strategy.py"
    plugin_path.write_text(source, encoding="utf-8")
    return plugin_path
