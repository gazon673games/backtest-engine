"""Load shared Workbench process settings from the repository-level JSON file."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

from backtest_engine.adapters.proxy import AdapterProxySettings
from backtest_engine.core.exceptions import StrategyPluginError


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    port_scan_limit: int = Field(default=100, ge=1)
    startup_timeout_ms: int = Field(default=120_000, ge=1_000)


class JobSettings(BaseModel):
    max_concurrency: int = Field(default=2, ge=1)


class PathSettings(BaseModel):
    data_root: Path = Path("data/workbench")
    runs_directory: str = "runs"
    datasets_directory: str = "datasets"
    packaged_data_directory: str = "workbench"


class DesktopSettings(BaseModel):
    width: int = Field(default=1440, ge=640)
    height: int = Field(default=920, ge=480)
    min_width: int = Field(default=980, ge=320)
    min_height: int = Field(default=680, ge=320)
    log_file: str = "workbench.log"


class StrategySettings(BaseModel):
    sdk_catalog_path: Path | None = None


class WorkbenchSettings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    jobs: JobSettings = Field(default_factory=JobSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    desktop: DesktopSettings = Field(default_factory=DesktopSettings)
    proxy: AdapterProxySettings = Field(default_factory=AdapterProxySettings)
    strategies: StrategySettings = Field(default_factory=StrategySettings)


def load_workbench_settings(path: Path | None = None) -> WorkbenchSettings:
    """Read explicit, environment, or repository settings; otherwise use safe defaults."""
    settings_path = path or configured_settings_path()
    if settings_path is not None and not settings_path.exists():
        raise FileNotFoundError(f"Workbench settings do not exist: {settings_path}")
    payload = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path else {}
    settings = WorkbenchSettings.model_validate(payload)
    override = os.environ.get("WORKBENCH_SDK_CATALOG_PATH")
    catalog = Path(override).expanduser() if override else settings.strategies.sdk_catalog_path
    if override == "":
        catalog = None
    if catalog is not None and not catalog.is_absolute():
        catalog = (settings_path.parent if settings_path else Path.cwd()) / catalog
    settings.strategies.sdk_catalog_path = catalog.absolute() if catalog is not None else None
    return settings


def configured_settings_path() -> Path | None:
    override = os.environ.get("WORKBENCH_SETTINGS_PATH")
    if override:
        return Path(override).expanduser().resolve()
    candidates = (
        Path.cwd() / "settings",
        Path(__file__).resolve().parents[4] / "settings",
        Path(__file__).resolve().parents[3] / "settings",
    )
    return next(
        (
            directory / filename
            for directory in candidates
            for filename in ("workbench.local.json", "workbench.json")
            if (directory / filename).exists()
        ),
        None,
    )


def runs_root(settings: WorkbenchSettings) -> Path:
    return settings.paths.data_root / settings.paths.runs_directory


def datasets_root(settings: WorkbenchSettings) -> Path:
    return settings.paths.data_root / settings.paths.datasets_directory


def configured_sdk_catalog() -> Path:
    root = load_workbench_settings().strategies.sdk_catalog_path
    if root is None:
        raise StrategyPluginError("SDK catalog is disabled; configure strategies.sdk_catalog_path")
    return root
