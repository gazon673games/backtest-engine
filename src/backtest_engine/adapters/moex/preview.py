"""Converts raw MOEX pages into a bounded progressive chart overview."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from backtest_engine.adapters.build_events import (
    DatasetBuildObserver,
    OhlcvPreviewReducer,
    preview_stride,
)
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.pipeline.paths import (
    default_continuous_symbol,
    source_timeframe,
)
from backtest_engine.adapters.moex.pipeline.stages import aggregate_bars, normalize_moex_candles

PREVIEW_COLUMNS = (
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "active_contract",
    "source_contract_symbol",
)


class MoexPreviewEmitter:
    def __init__(
        self,
        request: DatasetBuildRequest,
        observer: DatasetBuildObserver | None,
    ) -> None:
        self.request = request
        self.observer = observer
        self.preview_timeframe = (
            observer.preview_timeframe if observer is not None else request.timeframe
        )
        self.base_timeframe = source_timeframe(request.timeframe)
        self.pending_raw: list[dict[str, Any]] = []
        preview_request = request.model_copy(update={"timeframe": self.preview_timeframe})
        self.reducer = OhlcvPreviewReducer(preview_stride(preview_request))

    def consume(self, rows: list[dict[str, Any]]) -> None:
        if self.observer is None or not rows:
            return
        self.pending_raw.extend(rows)
        combined = deduplicate_raw_rows(self.pending_raw)
        prepared = self._prepare(combined)
        if prepared.height == 0:
            self.pending_raw = combined
            return
        cutoff = prepared.get_column("ts")[-1]
        completed = prepared.slice(0, max(0, prepared.height - 1))
        self.pending_raw = [row for row in combined if raw_timestamp(row) >= cutoff]
        self._emit(completed)

    def finish(self) -> None:
        if self.observer is None:
            return
        if self.pending_raw:
            self._emit(self._prepare(self.pending_raw))
            self.pending_raw = []
        final_rows = self.reducer.finish()
        if final_rows:
            self.observer.preview(final_rows, timeframe=self.preview_timeframe)

    def _prepare(self, rows: list[dict[str, Any]]) -> pl.DataFrame:
        normalized = normalize_moex_candles(
            pl.DataFrame(rows),
            continuous_symbol=(
                self.request.continuous_symbol or default_continuous_symbol(self.request)
            ),
            timeframe=self.base_timeframe,
        )
        if self.base_timeframe == self.preview_timeframe:
            return normalized
        return aggregate_bars(normalized, self.preview_timeframe)

    def _emit(self, frame: pl.DataFrame) -> None:
        if frame.height == 0 or self.observer is None:
            return
        columns = [column for column in PREVIEW_COLUMNS if column in frame.columns]
        rows = self.reducer.push(frame.select(columns).sort("ts").to_dicts())
        if rows:
            self.observer.preview(rows, timeframe=self.preview_timeframe)


def deduplicate_raw_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {(str(row.get("SECID", "")), raw_timestamp(row)): row for row in rows}
    return [keyed[key] for key in sorted(keyed, key=lambda item: item[1])]


def raw_timestamp(row: dict[str, Any]) -> datetime:
    raw = str(row.get("begin") or row.get("ts") or row.get("end"))
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return value.replace(tzinfo=None) if value.tzinfo is not None else value
