"""Exchange-neutral progress and provisional-preview contracts for dataset builds."""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Protocol

from backtest_engine.adapters.models import DatasetBuildRequest

PREVIEW_POINT_LIMIT = 12_000


class DatasetBuildObserver(Protocol):
    preview_timeframe: str

    def progress(self, *, stage: str, message: str, pct: float) -> None: ...

    def preview(self, rows: list[dict[str, Any]], *, timeframe: str) -> None: ...


class OhlcvPreviewReducer:
    """Preserve OHLC extrema while bounding a progressive overview by row groups."""

    def __init__(self, stride: int) -> None:
        self.stride = max(1, stride)
        self.pending: list[dict[str, Any]] = []

    def push(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.pending.extend(rows)
        complete_count = len(self.pending) // self.stride
        if complete_count == 0:
            return []
        consumed = complete_count * self.stride
        complete = self.pending[:consumed]
        self.pending = self.pending[consumed:]
        return [
            aggregate_ohlcv(complete[index : index + self.stride])
            for index in range(0, consumed, self.stride)
        ]

    def finish(self) -> list[dict[str, Any]]:
        if not self.pending:
            return []
        row = aggregate_ohlcv(self.pending)
        self.pending = []
        return [row]


def preview_stride(request: DatasetBuildRequest) -> int:
    start = date.fromisoformat(request.from_date)
    finish = date.fromisoformat(request.till_date)
    calendar_minutes = max(1, (finish - start).days + 1) * 24 * 60
    estimated_bars = math.ceil(calendar_minutes / timeframe_minutes(request.timeframe))
    return max(1, math.ceil(estimated_bars / PREVIEW_POINT_LIMIT))


def timeframe_minutes(timeframe: str) -> int:
    normalized = timeframe.strip().lower()
    if normalized == "1d":
        return 24 * 60
    if normalized.endswith("m") and normalized[:-1].isdigit():
        return max(1, int(normalized[:-1]))
    if normalized.endswith("h") and normalized[:-1].isdigit():
        return max(1, int(normalized[:-1]) * 60)
    raise ValueError(f"Unsupported preview timeframe: {timeframe}")


def aggregate_ohlcv(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    last = rows[-1]
    result = {
        "ts": first["ts"],
        "open": first["open"],
        "high": max(float(row["high"]) for row in rows),
        "low": min(float(row["low"]) for row in rows),
        "close": last["close"],
        "volume": sum(float(row.get("volume") or 0.0) for row in rows),
    }
    for column in ("active_contract", "source_contract_symbol"):
        if column in last:
            result[column] = last[column]
    return result
