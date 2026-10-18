from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

import polars as pl
from continuous_futures_builder.session_calendar import observe_sessions

from backtest_engine.core.exceptions import DatasetValidationError


@dataclass(frozen=True)
class MoexRegistryUpdateConfig:
    dataset_paths: list[Path]
    instrument_output_path: Path
    calendar_output_path: Path
    audit_output_path: Path
    price_currency: str = "RUB"
    pnl_currency: str = "RUB"
    contract_multiplier: float = 1.0
    lot_size: float = 1.0
    tick_size: float | None = None
    tick_value: float | None = None
    initial_margin: float | None = None


@dataclass(frozen=True)
class MoexRegistryUpdateResult:
    instrument_output_path: Path
    calendar_output_path: Path
    audit_output_path: Path
    instrument_count: int
    trading_day_count: int
    holiday_count: int
    short_day_count: int
    warnings: list[str]


def update_moex_registry_and_calendar(
    config: MoexRegistryUpdateConfig,
) -> MoexRegistryUpdateResult:
    if not config.dataset_paths:
        raise DatasetValidationError("dataset_paths must not be empty")
    frames = [_load_frame(path) for path in config.dataset_paths]
    combined = pl.concat(frames, how="diagonal_relaxed")
    if "ts" not in combined.columns or combined.height == 0:
        raise DatasetValidationError("MOEX registry update requires a non-empty ts column")
    combined = combined.with_columns(pl.col("ts").cast(pl.Datetime))
    warnings: list[str] = []
    instruments = _build_instruments(combined, config, warnings)
    calendar = _build_calendar(combined, warnings)
    audit = _build_audit(instruments, calendar, warnings)
    _write_json(config.instrument_output_path, instruments)
    _write_json(config.calendar_output_path, calendar)
    _write_json(config.audit_output_path, audit)
    return MoexRegistryUpdateResult(
        instrument_output_path=config.instrument_output_path,
        calendar_output_path=config.calendar_output_path,
        audit_output_path=config.audit_output_path,
        instrument_count=len(instruments["instruments"]),
        trading_day_count=len(calendar["trading_dates"]),
        holiday_count=len(calendar["holiday_dates"]),
        short_day_count=len(calendar["short_day_dates"]),
        warnings=warnings,
    )


def _build_instruments(
    frame: pl.DataFrame,
    config: MoexRegistryUpdateConfig,
    warnings: list[str],
) -> dict[str, Any]:
    symbol_columns = [
        column
        for column in ("source_contract_symbol", "active_contract", "symbol")
        if column in frame.columns
    ]
    if not symbol_columns:
        warnings.append("no instrument symbol columns found; using wildcard spec")
        symbols = ["*"]
    else:
        symbols = sorted(
            {
                str(value)
                for column in symbol_columns
                for value in frame.get_column(column).drop_nulls().unique().to_list()
                if str(value).strip()
            }
        )
    instruments = [
        {
            "symbol": symbol,
            "price_currency": config.price_currency,
            "pnl_currency": config.pnl_currency,
            "contract_multiplier": config.contract_multiplier,
            "lot_size": config.lot_size,
            "tick_size": config.tick_size,
            "tick_value": config.tick_value,
            "initial_margin": config.initial_margin,
        }
        for symbol in symbols
    ]
    return {
        "source": "moex_observed_datasets",
        "generated_at": datetime.now(UTC).isoformat(),
        "instruments": instruments,
    }


def _build_calendar(frame: pl.DataFrame, warnings: list[str]) -> dict[str, Any]:
    timestamps = [_as_datetime(value) for value in frame["ts"]]
    observed = observe_sessions(timestamps, main_start=time(10), main_end=time(18, 50))
    warnings.append("Observed calendar is diagnostic; supply independent expected sessions")
    return {
        "source": "moex_observed_datasets",
        "generated_at": datetime.now(UTC).isoformat(),
        "main_session_start": "10:00",
        "main_session_end": "18:50",
        "evening_session_start": "19:00",
        "evening_session_end": "23:50",
        **observed,
    }


def _build_audit(
    instruments: dict[str, Any],
    calendar: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    violations = []
    if not instruments["instruments"]:
        violations.append("instrument registry is empty")
    if not calendar["trading_dates"]:
        violations.append("calendar has no trading dates")
    return {
        "status": "warning" if not violations else "rejected",
        "source": "moex_observed_datasets",
        "generated_at": datetime.now(UTC).isoformat(),
        "instrument_count": len(instruments["instruments"]),
        "trading_day_count": len(calendar["trading_dates"]),
        "holiday_count": len(calendar["holiday_dates"]),
        "short_day_count": len(calendar["short_day_dates"]),
        "violations": violations,
        "warnings": warnings,
    }


def _load_frame(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise DatasetValidationError(f"Dataset path does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    raise DatasetValidationError(f"Unsupported MOEX dataset format: {path}")


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    raise DatasetValidationError(f"ts must contain datetime values, got {value!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
