from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from backtest_engine.core.exceptions import BacktestError, DatasetValidationError
from backtest_engine.data.session_model import (
    parse_date_value as parse_session_date_value,
)
from backtest_engine.data.session_model import (
    parse_time_value as parse_session_time_value,
)


def read_json_or_toml(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        msg = f"{label} file does not exist: {path}"
        raise DatasetValidationError(msg)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".toml":
        with path.open("rb") as spec_file:
            return tomllib.load(spec_file)
    msg = f"{label} must be JSON or TOML: {path}"
    raise DatasetValidationError(msg)


def as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    msg = f"ts must contain datetime values, got {value!r}"
    raise DatasetValidationError(msg)


def parse_date_value(value: Any) -> date:
    try:
        return parse_session_date_value(value)
    except BacktestError as exc:
        msg = str(exc)
        raise DatasetValidationError(msg) from exc


def parse_time_value(value: Any) -> time:
    try:
        return parse_session_time_value(value)
    except BacktestError as exc:
        msg = str(exc)
        raise DatasetValidationError(msg) from exc


def to_iso(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def unique_sorted_times(values: list[time]) -> list[time]:
    return sorted(set(values))


def ratio_float(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0.0 else 0.0


def frame_value_to_iso(frame: Any, column: str, index: int) -> str | None:
    if frame.height == 0:
        return None
    return to_iso(frame.get_column(column)[index])


def decision_times_text(values: list[time]) -> str | None:
    if not values:
        return None
    return ",".join(value.isoformat(timespec="minutes") for value in values)


def decision_time_text_from_value(value: time | None) -> str | None:
    if value is None:
        return None
    return value.isoformat(timespec="minutes")


def json_dumps(payload: Any) -> str | None:
    if not payload:
        return None
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
