"""Fits deterministic execution assumptions from observed order fills."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from backtest_engine.core.exceptions import DatasetValidationError


class ExecutionCalibration(BaseModel):
    schema_version: int = 1
    sample_count: int = Field(gt=0)
    median_spread_bps: float = Field(ge=0.0)
    median_signed_slippage_bps: float
    median_market_impact_bps: float = Field(ge=0.0)
    p95_market_impact_bps: float = Field(ge=0.0)
    median_latency_ms: float = Field(ge=0.0)
    p95_participation: float = Field(gt=0.0, le=1.0)
    source_sha256: str | None = None

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def calibrate_execution(
    frame: pl.DataFrame,
    *,
    source_sha256: str | None = None,
) -> ExecutionCalibration:
    required = {
        "side",
        "arrival_mid",
        "fill_price",
        "order_quantity",
        "market_volume",
        "submitted_ts",
        "filled_ts",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DatasetValidationError("Execution fills are missing columns: " + ", ".join(missing))
    signed_slippage: list[float] = []
    spread: list[float] = []
    latency: list[float] = []
    participation: list[float] = []
    for index, row in enumerate(frame.iter_rows(named=True)):
        side = str(row["side"]).strip().lower()
        if side not in {"buy", "sell"}:
            raise DatasetValidationError(f"Invalid fill side at row {index}: {side}")
        midpoint = positive(row["arrival_mid"], "arrival_mid", index)
        fill_price = positive(row["fill_price"], "fill_price", index)
        quantity = positive(row["order_quantity"], "order_quantity", index)
        volume = positive(row["market_volume"], "market_volume", index)
        direction = 1.0 if side == "buy" else -1.0
        signed_slippage.append(direction * (fill_price / midpoint - 1.0) * 10_000.0)
        bid = optional_positive(row.get("arrival_bid"))
        ask = optional_positive(row.get("arrival_ask"))
        if bid is not None and ask is not None:
            if bid > ask:
                raise DatasetValidationError(f"arrival_bid exceeds arrival_ask at row {index}")
            spread.append((ask - bid) / midpoint * 10_000.0)
        latency.append(max(timestamp_ms(row["filled_ts"]) - timestamp_ms(row["submitted_ts"]), 0.0))
        participation.append(min(quantity / volume, 1.0))
    if not signed_slippage:
        raise DatasetValidationError("Execution fills dataset is empty")
    median_spread = percentile(spread, 0.5) or 0.0
    residual = [max(value - median_spread / 2.0, 0.0) for value in signed_slippage]
    return ExecutionCalibration(
        sample_count=len(signed_slippage),
        median_spread_bps=median_spread,
        median_signed_slippage_bps=percentile(signed_slippage, 0.5) or 0.0,
        median_market_impact_bps=percentile(residual, 0.5) or 0.0,
        p95_market_impact_bps=percentile(residual, 0.95) or 0.0,
        median_latency_ms=percentile(latency, 0.5) or 0.0,
        p95_participation=percentile(participation, 0.95) or 1.0,
        source_sha256=source_sha256,
    )


def calibrate_execution_path(path: Path) -> ExecutionCalibration:
    frame = read_frame(path)
    return calibrate_execution(frame, source_sha256=file_sha256(path))


def load_execution_calibration(path: Path) -> ExecutionCalibration:
    if not path.is_file():
        raise DatasetValidationError(f"Execution calibration file does not exist: {path}")
    try:
        return ExecutionCalibration.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DatasetValidationError(f"Invalid execution calibration: {path}: {exc}") from exc


def calibration_updates(path: Path, bar_minutes: int) -> dict[str, Any]:
    calibration = load_execution_calibration(path)
    latency_bars = math.ceil(calibration.median_latency_ms / (bar_minutes * 60_000.0))
    return {
        "bid_ask_spread_bps": calibration.median_spread_bps,
        "market_impact_bps": calibration.median_market_impact_bps,
        "latency_bars": latency_bars,
        "volume_participation_limit": calibration.p95_participation,
    }


def read_frame(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    raise DatasetValidationError("Execution fills must be CSV or Parquet")


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def positive(value: Any, field: str, index: int) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise DatasetValidationError(f"{field} must be positive at row {index}")
    return parsed


def optional_positive(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def timestamp_ms(value: Any) -> float:
    if isinstance(value, datetime):
        return value.timestamp() * 1000.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000.0
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().timestamp() * 1000.0
    raise DatasetValidationError(f"Unsupported execution timestamp: {value!r}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
