"""Persist both successful and failed cases through one execution boundary."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from backtest_engine.core.file_io import write_json_atomic


def execute_case_record(
    *, name: str, fingerprint: str, checkpoint: Path, operation: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    try:
        record = {"name": name, "status": "completed", "fingerprint": fingerprint, **operation()}
    except Exception as error:
        record = {
            "name": name,
            "status": "failed",
            "fingerprint": fingerprint,
            "error": f"{type(error).__name__}: {error}",
        }
    write_json_atomic(checkpoint, record)
    return record
