from __future__ import annotations

import hashlib
import json
import math
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import BacktestError


def finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise BacktestError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise BacktestError(f"{name} must be finite")
    return parsed


def as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    raise BacktestError(f"ts must be datetime-like, got {value!r}")


def read_json_or_toml(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise BacktestError(f"{label} file does not exist: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".toml":
        with path.open("rb") as payload_file:
            return tomllib.load(payload_file)
    raise BacktestError(f"{label} must be JSON or TOML: {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def max_drawdown(equity_values: list[float]) -> float:
    peak = equity_values[0]
    drawdown = 0.0
    for value in equity_values:
        peak = max(peak, value)
        if peak != 0.0:
            drawdown = min(drawdown, value / peak - 1.0)
    return drawdown


def sharpe(returns: list[float], periods_per_year: float | None) -> float | None:
    if periods_per_year is None or len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    volatility = math.sqrt(variance)
    if volatility == 0.0:
        return None
    return mean / volatility * math.sqrt(periods_per_year)


def count_value(frame: pl.DataFrame, column: str, expected: str) -> int:
    if frame.height == 0 or column not in frame.columns:
        return 0
    return sum(1 for value in frame.get_column(column).to_list() if value == expected)
