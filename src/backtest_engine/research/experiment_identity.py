"""Stable identities for reproducible experiments and cached runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from strategy_sdk import __version__ as sdk_version

from backtest_engine.intraday.event.config import IntradayEventBacktestConfig
from backtest_engine.intraday.event.utils import file_sha256
from backtest_engine.strategies.plugins.loader import strategy_file_sha256


def canonical_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def event_run_identity(
    config: IntradayEventBacktestConfig,
    dataset_validation: dict[str, Any],
) -> dict[str, Any]:
    """Return every material input needed to decide whether a run is reusable."""
    return {
        "identity_version": 1,
        "engine": "intraday_event_v1",
        "dataset": {
            "path": str(config.dataset.path.resolve()),
            "sha256": file_sha256(config.dataset.path),
            "validation": dataset_validation,
            "catalog": artifact_identity(config.dataset.catalog_path),
            "quality_approval": artifact_identity(config.dataset.quality_approval_path),
        },
        "strategy": {
            "sdk_version": sdk_version,
            "path": str(config.strategy.strategy_file.resolve()),
            "sha256": strategy_file_sha256(config.strategy.strategy_file),
            "params": dict(config.strategy.strategy_params),
        },
        "instrument": config.instrument.model_dump(mode="json"),
        "portfolio": config.portfolio.model_dump(mode="json"),
        "session": config.session.model_dump(mode="json"),
        "execution": config.execution.model_dump(mode="json"),
        "output_contract": {
            "write_csv": config.output.write_csv,
        },
        "input_artifacts": event_input_artifacts(config),
    }


def event_run_fingerprint(
    config: IntradayEventBacktestConfig,
    dataset_validation: dict[str, Any],
) -> str:
    return canonical_fingerprint(event_run_identity(config, dataset_validation))


def event_input_artifacts(config: IntradayEventBacktestConfig) -> dict[str, Any]:
    paths = {
        "catalog": config.dataset.catalog_path,
        "quality_approval": config.dataset.quality_approval_path,
        "instrument_registry": config.instrument.registry_path,
        "instrument_spec": config.instrument.spec_path,
        "calendar": config.session.calendar_path,
        "valuation_rates": config.portfolio.valuation_rates_path,
        "execution_calibration": config.execution.calibration_path,
        "order_book": config.execution.order_book_path,
        "reconciliation_statement": config.portfolio.reconciliation_statement_path,
    }
    return {name: artifact_identity(path) for name, path in paths.items()}


def artifact_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "sha256": file_sha256(resolved) if resolved.is_file() else None,
    }
