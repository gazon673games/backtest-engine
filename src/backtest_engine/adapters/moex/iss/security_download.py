"""Reuse local MOEX observations and fetch only uncovered date intervals."""

from __future__ import annotations

from typing import Any

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.candle_coverage import missing_intervals
from backtest_engine.adapters.moex.iss.candle_transport import (
    RawRowsCallback,
    download_network_candles,
    enrich_candle_rows,
    live_tail_requested,
    merge_candle_rows,
    require_candles,
)
from backtest_engine.adapters.moex.iss.raw_snapshot_cache import (
    available_candle_snapshots,
    save_candle_snapshot,
)


def download_security_candles(
    request: DatasetBuildRequest,
    on_rows: RawRowsCallback | None = None,
) -> list[dict[str, Any]]:
    snapshots = (
        available_candle_snapshots(request) if request.cache_policy in {"prefer", "offline"} else []
    )
    if not snapshots:
        if request.cache_policy == "offline":
            raise FileNotFoundError(f"MOEX candle snapshot not found for {request.security}")
        rows = download_network_candles(request, on_rows=on_rows)
        if request.cache_policy != "network":
            save_candle_snapshot(request, rows)
        return rows
    cached = merge_candle_rows([], [row for snapshot in snapshots for row in snapshot.rows])
    cached = enrich_candle_rows(cached, request)
    if request.cache_policy == "offline":
        return publish(require_candles(cached, request, source="snapshot"), on_rows)
    missing = missing_intervals(request, snapshots)
    for first, last in missing:
        child = request.model_copy(update={"from_date": first, "till_date": last})
        try:
            fresh = download_network_candles(child, allow_empty=True)
        except (OSError, ValueError):
            if live_tail_requested(request):
                raise
            return publish(require_candles(cached, request, source="snapshot"), on_rows)
        save_candle_snapshot(child, fresh)
        cached = merge_candle_rows(cached, fresh)
    rows = require_candles(cached, request, source="MOEX")
    if missing:
        save_candle_snapshot(request, rows)
    return publish(rows, on_rows)


def publish(rows: list[dict[str, Any]], callback: RawRowsCallback | None) -> list[dict[str, Any]]:
    if rows and callback is not None:
        callback(rows)
    return rows
