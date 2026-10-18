"""Stores one immutable raw payload per published MOEX dataset version."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from backtest_engine.adapters.raw_json import (
    encode_json,
    gzip_matches,
    write_gzip_bytes,
)


def write_raw_snapshot(raw_dir: Path, payload: Any) -> Path:
    contents = encode_json(payload, sort_keys=True)
    encoded = contents.removesuffix(b"\n")
    digest = hashlib.sha256(encoded).hexdigest()
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{digest}.json.gz"
    if path.exists():
        ensure_snapshot_contents(path, contents)
        return path
    write_gzip_bytes(path, contents)
    return path


def ensure_snapshot_contents(path: Path, expected: bytes) -> None:
    matches = (
        gzip_matches(path, expected)
        if path.name.lower().endswith(".gz")
        else path.read_bytes() == expected
    )
    if not matches:
        raise ValueError(f"Raw snapshot hash collision or corrupt file: {path}")
