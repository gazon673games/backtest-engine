"""Content identity for dataset, strategy and effective case parameters."""

import hashlib
import json
from pathlib import Path
from typing import Any

from backtest_engine.data.store import file_sha256


def case_identity(dataset_path: Path, strategy_file: Path, parameters: dict[str, Any]) -> str:
    payload = {
        "dataset_sha256": file_sha256(dataset_path),
        "strategy_sha256": file_sha256(strategy_file),
        "parameters": parameters,
    }
    from backtest_engine.core.process_settings import configured_sdk_catalog
    from backtest_engine.strategies.sdk_catalog_sources import catalog_reference, catalog_snapshot

    reference = (
        catalog_reference(strategy_file.read_text(encoding="utf-8-sig"))
        if strategy_file.suffix == ".py"
        else None
    )
    if reference is not None:
        payload["catalog_snapshot"] = catalog_snapshot(configured_sdk_catalog(), reference)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
