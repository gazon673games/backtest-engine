"""Deterministic compressed storage for auditable adapter JSON payloads."""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path
from typing import Any


def encode_json(payload: Any, *, sort_keys: bool = False, indent: int | None = None) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=sort_keys,
        indent=indent,
        default=str,
    )
    return (text + "\n").encode("utf-8")


def write_gzip_bytes(path: Path, contents: bytes) -> None:
    """Atomically write reproducible gzip bytes (mtime is fixed at zero)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=6,
            fileobj=handle,
            mtime=0,
        ) as target:
            for offset in range(0, len(contents), 1024 * 1024):
                target.write(contents[offset : offset + 1024 * 1024])
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_gzip_json(
    path: Path,
    payload: Any,
    *,
    sort_keys: bool = False,
    indent: int | None = None,
) -> None:
    write_gzip_bytes(path, encode_json(payload, sort_keys=sort_keys, indent=indent))


def read_json(path: Path) -> Any:
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            return json.load(source)
    return json.loads(path.read_text(encoding="utf-8"))


def read_gzip_bytes(path: Path) -> bytes:
    with gzip.open(path, "rb") as source:
        return source.read()


def gzip_matches(path: Path, expected: bytes) -> bool:
    offset = 0
    with gzip.open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            if chunk != expected[offset : offset + len(chunk)]:
                return False
            offset += len(chunk)
    return offset == len(expected)
