from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class AdapterCapabilities(BaseModel):
    asset_classes: list[str] = Field(default_factory=lambda: ["generic"])
    data_kinds: list[str] = Field(default_factory=lambda: ["bars"])
    instrument_modes: list[str] = Field(default_factory=lambda: ["security"])
    default_engine: str | None = None
    default_instrument_mode: str = "security"
    instrument_catalogs: dict[str, str] = Field(default_factory=dict)
    timeframes: list[str] = Field(default_factory=lambda: ["1m", "5m", "15m", "60m"])
    cache_policies: list[str] = Field(default_factory=list)
    archive_fallback_policies: list[str] = Field(default_factory=list)
    default_archive_fallback_policy: str = "missing_only"
    roll_rules: list[str] = Field(default_factory=list)
    supports_instrument_search: bool = True
    supports_dataset_build: bool = True
    supports_offline: bool = False
    default_delay_minutes: int | None = Field(default=None, ge=0)
    provider_supports_realtime: bool = False
    realtime_requires_subscription: bool = False
    adapter_realtime_enabled: bool = False
    freshness_ttl_minutes: int | None = Field(default=None, ge=1)
    session_windows: list[dict[str, str]] = Field(default_factory=list)
    execution_defaults: dict[str, float | str] = Field(default_factory=dict)
    supports_rule_archive: bool = False
    supports_history_range: bool = False
    supports_dataset_prefix_reuse: bool = False
    default_rules_enforcement: Literal["audit", "strict"] = "strict"
    default_quality_approval_required: bool = True
    funding_price_policies: list[str] = Field(default_factory=list)


class AdapterInfo(BaseModel):
    id: str
    name: str
    description: str
    capabilities: AdapterCapabilities = Field(default_factory=AdapterCapabilities)


class AdapterItem(BaseModel):
    id: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class InstrumentItem(BaseModel):
    symbol: str
    short_name: str | None = None
    name: str | None = None
    board: str | None = None
    market: str | None = None
    engine: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InstrumentHistoryRange(BaseModel):
    adapter_id: str
    engine: str
    symbol: str
    timeframe: str
    first_bar: datetime
    last_bar: datetime
    from_date: date
    till_date: date
    timezone: str = "UTC"
    start_policy: str = "first_complete_day"
    retrieved_at: datetime
    components: dict[str, dict[str, str]] = Field(default_factory=dict)


class DatasetPrefixInput(BaseModel):
    dataset_id: str
    dataset_path: Path
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    adapter_id: str
    engine: str
    symbol: str
    timeframe: str
    quality_status: Literal["approved", "warning"]
    provenance: dict[str, Any] = Field(default_factory=dict)


class DatasetBuildRequest(BaseModel):
    adapter_id: str
    engine: str | None = None
    market: str | None = None
    board: str | None = None
    security: str
    instrument_mode: str = "security"
    asset_class: str | None = None
    data_kind: str = "bars"
    from_date: str
    till_date: str
    timeframe: str = "5m"
    dataset_id: str | None = None
    continuous_symbol: str | None = None
    output_root: Path = Path("data/workbench/datasets")
    gap_policy: str = "audit"
    gap_repair_policy: str = "none"
    cache_policy: str = "prefer"
    cache_dir: Path | None = None
    roll_rule: str | None = None
    source_delay_minutes: int | None = Field(default=None, ge=0)
    refresh_live_tail: bool = True
    source_path: Path | None = None
    reuse_dataset: DatasetPrefixInput | None = None
    calendar_path: Path | None = None
    rules_path: Path | None = None
    rules_enforcement: str = "audit"
    history_source: str = "auto"
    funding_price_policy: Literal["event_only", "minute_mark_open_estimate"] = "event_only"
    archive_fallback_policy: Literal[
        "missing_only", "invalid_timestamps", "invalid_timestamps_and_missing_marks"
    ] = "missing_only"


class DatasetBuildResult(BaseModel):
    adapter_id: str
    dataset_id: str
    dataset_path: Path
    output_root: Path
    approval_status: str
    approval_path: Path
    manifest_path: Path
    raw_snapshot_path: Path
    timeframe: str
    row_count: int
    quality: dict[str, Any] = Field(default_factory=dict)
    instrument_registry_path: Path | None = None
    instrument_registry_audit_path: Path | None = None
    calendar_path: Path | None = None
    calendar_audit_path: Path | None = None
    catalog_path: Path | None = None
    source_as_of: str | None = None
    finalized_through: str | None = None
    bar_time_semantics: str | None = None
