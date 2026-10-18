from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backtest_engine.adapters.moex.sessions import MOEX_SESSION_WINDOWS


@dataclass(frozen=True)
class MoexDataPipelineConfig:
    output_root: Path
    raw_snapshot_path: Path | None = None
    source_url: str | None = None
    dataset_id: str = "moex:dataset"
    continuous_symbol: str = "MOEX_NEAREST"
    base_timeframe: str = "1m"
    aggregate_timeframes: list[str] = field(default_factory=lambda: ["5m"])
    gap_policy: str = "reject"
    gap_repair_policy: str = "none"
    roll_gap_policy: str = "audit"
    max_gap_ratio: float = 0.005
    max_roll_gap_pct: float = 0.10
    source_delay_minutes: int = 15
    source_as_of: datetime | None = None
    session_windows: tuple[tuple[str, str, str], ...] = MOEX_SESSION_WINDOWS


@dataclass(frozen=True)
class MoexDataPipelineResult:
    output_root: Path
    dataset_id: str
    raw_snapshot_path: Path
    normalized_path: Path
    nearest_path: Path
    continuous_path: Path
    manifest_path: Path
    approval_path: Path
    gap_audit_path: Path
    gap_repair_path: Path
    roll_audit_path: Path
    aggregate_paths: dict[str, Path]
    approval_status: str
    source_as_of: str
    finalized_through: str
