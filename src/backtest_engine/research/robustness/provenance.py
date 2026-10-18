"""Hashes for immutable plans, native artifacts and dirty runtime source trees."""

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import ohlcv_data_store
import strategy_sdk

import backtest_engine

RUNTIME_PACKAGES = (
    backtest_engine,
    strategy_sdk,
    ohlcv_data_store,
)

INPUT_FIELDS = (
    "dataset_path",
    "quality_approval_path",
    "instrument_registry_path",
    "calendar_path",
    "catalog_path",
    "valuation_rates_path",
)
ARTIFACT_NAMES = (
    "config.json",
    "metrics.json",
    "events.parquet",
    "portfolio_ledger.parquet",
    "trades.parquet",
    "quality_approval.json",
    "catalog_approval.json",
    "strategy_snapshot.py",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def request_input_hashes(request: dict[str, Any]) -> dict[str, str]:
    return {
        str(Path(request[key]).resolve()): file_hash(Path(request[key]))
        for key in INPUT_FIELDS
        if request.get(key)
    }


def runtime_identity() -> dict[str, Any]:
    sources = {}
    for package in RUNTIME_PACKAGES:
        root = Path(package.__file__).resolve().parent
        for path in sorted(root.rglob("*.py")):
            sources[f"{package.__name__}/{path.relative_to(root).as_posix()}"] = file_hash(path)
    identity = {
        "source_hashes": sources,
        "python": sys.version,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    # PyInstaller embeds Python bytecode inside the executable instead of source files.
    if getattr(sys, "frozen", False):
        identity["frozen_executable_sha256"] = file_hash(Path(sys.executable))
    return identity


def artifact_hashes(path: Path) -> dict[str, str]:
    return {name: file_hash(path / name) for name in ARTIFACT_NAMES if (path / name).is_file()}


def verify_hashes(records: dict[str, str], root: Path | None = None) -> None:
    for name, expected in records.items():
        path = root / name if root is not None else Path(name)
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"pinned input or native result changed: {path}")
