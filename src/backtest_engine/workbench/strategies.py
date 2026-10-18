from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from strategy_sdk import feature_snapshot

from backtest_engine.strategies.runtime import StrategyParamSpec, validate_strategy_params
from backtest_engine.workbench.sdk_catalog import freeze_catalog_source
from backtest_engine.workbench.strategy_contract import load_workbench_strategy_metadata


def strategies_root(output_root: Path) -> Path:
    return output_root.parent / "strategies"


def list_saved_strategies(root: Path) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    records = [read_strategy(path) for path in sorted(root.glob("*.json"))]
    return sorted(records, key=lambda item: str(item.get("updated_at", "")), reverse=True)


def read_strategy(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_saved_strategy(root: Path, strategy_id: str) -> dict[str, Any]:
    path = root / f"{safe_strategy_id(strategy_id)}.json"
    if not path.exists():
        raise FileNotFoundError(f"strategy not found: {strategy_id}")
    return read_strategy(path)


def save_strategy(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    source = str(payload.get("source", ""))
    if not source.strip():
        raise ValueError("strategy source is required")
    requested_id = payload.get("id")
    strategy_id = (
        safe_strategy_id(str(requested_id))
        if requested_id
        else available_strategy_id(root, str(payload.get("name") or "strategy"))
    )
    path = root / f"{strategy_id}.json"
    previous = read_strategy(path) if path.exists() else {}
    now = datetime.now(UTC).isoformat()
    version = int(previous.get("version", 0)) + 1
    language = payload.get("language", previous.get("language", "javascript"))
    raw_params = payload.get("params", previous.get("params", {}))
    metadata = metadata_from_source(source, root, language=language, params=raw_params)
    record = {
        "id": strategy_id,
        "name": str(payload.get("name") or previous.get("name") or strategy_id),
        "tags": list(payload.get("tags") or previous.get("tags") or []),
        "source": source,
        "language": language,
        "metadata": metadata,
        "params": normalized_strategy_params(metadata, raw_params),
        "version": version,
        "created_at": previous.get("created_at", now),
        "updated_at": now,
        "versions": [*previous.get("versions", []), version_record(previous, now)],
    }
    record = feature_snapshot(**record)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def normalized_strategy_params(metadata: dict[str, Any], raw_params: Any) -> dict[str, Any]:
    if not isinstance(raw_params, dict):
        raise ValueError("strategy params must be an object")
    specs = tuple(StrategyParamSpec(**item) for item in metadata.get("params", ()))
    return validate_strategy_params(raw_params, specs)


def metadata_from_source(
    source: str, root: Path, *, language="javascript", params=None
) -> dict[str, Any]:
    if language not in {"python", "javascript"}:
        raise ValueError("strategy language must be python or javascript")
    if len(source.encode("utf-8")) > 256 * 1024:
        raise ValueError("strategy source is too large")
    temp_dir = root / ".metadata"
    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".py" if language == "python" else ".js"
    temp_path = temp_dir / f"{strategy_digest(source)}{suffix}"
    temp_path.write_text(freeze_catalog_source(source, language), encoding="utf-8")
    metadata = load_workbench_strategy_metadata(temp_path, params)
    return feature_snapshot(**asdict(metadata))


def version_record(previous: dict[str, Any], now: str) -> dict[str, Any]:
    if not previous:
        return {"version": 1, "saved_at": now}
    return {
        "version": previous.get("version", 1),
        "saved_at": previous.get("updated_at", now),
        "source_sha256": strategy_digest(str(previous.get("source", ""))),
    }


def strategy_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def available_strategy_id(root: Path, name: str) -> str:
    base = safe_strategy_id(name)
    candidate = base
    suffix = 2
    while (root / f"{candidate}.json").exists():
        candidate = safe_strategy_id(f"{base[:75]}-{suffix}")
        suffix += 1
    return candidate


def safe_strategy_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    safe = safe.strip("._-")
    if not safe:
        raise ValueError("strategy id is empty")
    return safe[:80]
