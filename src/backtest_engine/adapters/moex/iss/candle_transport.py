"""Pages through MOEX ISS candle responses and normalizes transport rows."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.client import (
    ISS_BASE_URL,
    cursor_total,
    fetch_iss_json,
    iss_table_rows,
)
from backtest_engine.adapters.moex.pipeline.paths import quote
from backtest_engine.adapters.moex.sessions import MOEX_TIMEZONE

MOEX_NATIVE_INTERVALS = {"1m": 1, "10m": 10, "60m": 60, "1d": 24}
RawRowsCallback = Callable[[list[dict[str, Any]]], None]


def download_network_candles(
    request: DatasetBuildRequest,
    on_rows: RawRowsCallback | None = None,
    *,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    interval = MOEX_NATIVE_INTERVALS.get(request.timeframe, 1)
    url = (
        f"{ISS_BASE_URL}/engines/{quote(request.engine)}/markets/{quote(request.market)}"
        f"/securities/{quote(request.security)}/candles.json"
    )
    rows: list[dict[str, Any]] = []
    start = 0
    last_page_start = 0
    pending_page: list[dict[str, Any]] = []
    while True:
        last_page_start = start
        payload = fetch_candle_page(request, url, interval, start, request.cache_policy)
        page = iss_table_rows(payload, "candles")
        if not page:
            break
        enriched = enrich_candle_rows(page, request)
        if pending_page and on_rows is not None:
            on_rows(pending_page)
        pending_page = enriched
        rows.extend(enriched)
        start += len(page)
        total = cursor_total(payload, "candles.cursor")
        if total is not None and start >= total:
            break
        if total is None and len(page) < 500:
            break
    if request.cache_policy == "prefer" and live_tail_requested(request):
        fresh = download_candle_pages(
            request,
            url,
            interval,
            start=last_page_start,
            cache_policy="refresh",
        )
        if fresh:
            del rows[last_page_start:]
            rows.extend(fresh)
            publish_pages(fresh, on_rows)
        elif pending_page and on_rows is not None:
            on_rows(pending_page)
    elif pending_page and on_rows is not None:
        on_rows(pending_page)
    if not rows and not allow_empty:
        raise ValueError(f"MOEX returned no candles for {request.security}")
    return rows


def publish_pages(rows: list[dict[str, Any]], on_rows: RawRowsCallback | None) -> None:
    if on_rows is None:
        return
    for offset in range(0, len(rows), 500):
        on_rows(rows[offset : offset + 500])


def download_candle_pages(
    request: DatasetBuildRequest,
    url: str,
    interval: int,
    *,
    start: int,
    cache_policy: str,
) -> list[dict[str, Any]]:
    rows = []
    while True:
        payload = fetch_candle_page(request, url, interval, start, cache_policy)
        page = iss_table_rows(payload, "candles")
        if not page:
            break
        rows.extend(enrich_candle_rows(page, request))
        start += len(page)
        total = cursor_total(payload, "candles.cursor")
        if (total is not None and start >= total) or (total is None and len(page) < 500):
            break
    return rows


def fetch_candle_page(
    request: DatasetBuildRequest,
    url: str,
    interval: int,
    start: int,
    cache_policy: str,
) -> dict[str, Any]:
    return fetch_iss_json(
        url,
        {
            "from": request.from_date,
            "till": request.till_date,
            "interval": str(interval),
            "start": str(start),
            "limit": "500",
            "iss.meta": "off",
            "candles.columns": "begin,end,open,high,low,close,volume,value",
        },
        cache_policy=cache_policy,
        cache_dir=request.cache_dir,
    )


def merge_candle_rows(
    cached: list[dict[str, Any]],
    fresh: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_timestamp = {
        str(row.get("begin") or row.get("ts") or row.get("end")): row for row in [*cached, *fresh]
    }
    return [by_timestamp[key] for key in sorted(by_timestamp)]


def require_candles(
    rows: list[dict[str, Any]],
    request: DatasetBuildRequest,
    *,
    source: str,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError(f"{source} has no candles for {request.security}")
    return rows


def enrich_candle_rows(
    rows: list[dict[str, Any]],
    request: DatasetBuildRequest,
) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "SECID": request.security,
                "source": "moex_iss",
                "engine": request.engine,
                "market": request.market,
                "board": request.board,
                "currency": "RUB",
            }
        )
        enriched.append(item)
    return enriched


def live_tail_requested(request: DatasetBuildRequest) -> bool:
    today = datetime.now(MOEX_TIMEZONE).date().isoformat()
    return request.refresh_live_tail and request.till_date >= today
