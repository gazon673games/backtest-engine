"""Point-in-time FX and collateral rates with strict no-lookahead lookup."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError


@dataclass(frozen=True)
class HistoricalRate:
    timestamp: float
    source_currency: str
    target_currency: str
    rate: float


class HistoricalRateBook:
    def __init__(self, rates: list[HistoricalRate]) -> None:
        grouped: dict[tuple[str, str], list[HistoricalRate]] = {}
        for rate in rates:
            grouped.setdefault((rate.source_currency, rate.target_currency), []).append(rate)
        self._rates = {
            pair: sorted(items, key=lambda item: item.timestamp) for pair, items in grouped.items()
        }
        self._timestamps = {
            pair: [item.timestamp for item in items] for pair, items in self._rates.items()
        }

    @classmethod
    def from_path(cls, path: Path) -> HistoricalRateBook:
        if not path.is_file():
            raise DatasetValidationError(f"Historical rates file does not exist: {path}")
        try:
            if path.suffix == ".csv":
                frame = pl.read_csv(path, try_parse_dates=True)
            else:
                frame = pl.read_parquet(path)
        except Exception as exc:
            raise DatasetValidationError(f"Failed to read historical rates: {path}") from exc
        return cls(parse_rate_frame(frame))

    def rates_at(
        self,
        ts: datetime,
        static_rates: dict[str, float] | None = None,
    ) -> dict[str, float]:
        result = dict(static_rates or {})
        timestamp = unix_timestamp(ts)
        for pair, items in self._rates.items():
            index = bisect_right(self._timestamps[pair], timestamp) - 1
            if index >= 0:
                result[f"{pair[0]}/{pair[1]}"] = items[index].rate
        return result


def parse_rate_frame(frame: pl.DataFrame) -> list[HistoricalRate]:
    aliases = {
        "base_currency": "source_currency",
        "quote_currency": "target_currency",
    }
    for source, target in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame = frame.rename({source: target})
    required = {"ts", "source_currency", "target_currency", "rate"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DatasetValidationError("Historical rates are missing columns: " + ", ".join(missing))
    result = []
    seen: set[tuple[float, str, str]] = set()
    for row in frame.select(sorted(required)).iter_rows(named=True):
        ts = as_datetime(row["ts"])
        source = str(row["source_currency"]).strip().upper()
        target = str(row["target_currency"]).strip().upper()
        rate = float(row["rate"])
        if not source or not target or not math.isfinite(rate) or rate <= 0.0:
            raise DatasetValidationError(f"Invalid historical rate row: {row}")
        key = (unix_timestamp(ts), source, target)
        if key in seen:
            raise DatasetValidationError(
                f"Duplicate historical rate at {ts.isoformat()}: {source}/{target}"
            )
        seen.add(key)
        result.append(HistoricalRate(key[0], source, target, rate))
    if not result:
        raise DatasetValidationError("Historical rates dataset is empty")
    return result


def as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    raise DatasetValidationError(f"Historical rate ts must be datetime, got {value!r}")


def unix_timestamp(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.timestamp()
