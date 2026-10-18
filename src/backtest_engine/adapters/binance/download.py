"""Paginated Binance kline and funding downloads."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from backtest_engine.adapters.binance.funding import ordered_funding_events
from backtest_engine.adapters.binance.transport import BinanceTransport, endpoint
from backtest_engine.adapters.build_events import DatasetBuildObserver
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.timeframes import parse_timeframe_minutes

BINANCE_TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")


def request_timeframe(timeframe: str) -> str:
    normalized = timeframe.strip()
    if normalized.endswith("m") and normalized not in BINANCE_TIMEFRAMES:
        minutes = parse_timeframe_minutes(normalized)
        normalized = (
            "1d" if minutes == 1440 else f"{minutes // 60}h" if minutes % 60 == 0 else normalized
        )
    if normalized not in BINANCE_TIMEFRAMES:
        raise DatasetValidationError(f"Unsupported Binance timeframe: {timeframe}")
    return normalized


def canonical_timeframe(timeframe: str) -> str:
    native = request_timeframe(timeframe)
    if native.endswith("h"):
        return f"{int(native[:-1]) * 60}m"
    return "1440m" if native == "1d" else native


def request_bounds(from_date: str, till_date: str) -> tuple[int, int]:
    start = datetime.combine(date.fromisoformat(from_date), time.min, tzinfo=UTC)
    finish = datetime.combine(
        date.fromisoformat(till_date) + timedelta(days=1),
        time.min,
        tzinfo=UTC,
    ) - timedelta(milliseconds=1)
    if finish < start:
        raise DatasetValidationError("Binance till_date must not precede from_date")
    return int(start.timestamp() * 1000), int(finish.timestamp() * 1000)


def download_klines(
    transport: BinanceTransport,
    *,
    product: str,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    observer: DatasetBuildObserver | None = None,
    progress_range: tuple[float, float] = (0.12, 0.62),
) -> list[list[Any]]:
    result: list[list[Any]] = []
    cursor = start_ms
    page_count = 0
    while cursor <= end_ms:
        page = transport.get(
            product,
            endpoint(product, "klines"),
            {
                "symbol": symbol,
                "interval": request_timeframe(timeframe),
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not isinstance(page, list) or not page:
            break
        result.extend(page)
        next_cursor = int(page[-1][6]) + 1
        if next_cursor <= cursor:
            raise DatasetValidationError("Binance kline pagination did not advance")
        cursor = next_cursor
        page_count += 1
        publish_download_progress(
            observer, symbol, "candles", page_count, cursor, start_ms, end_ms, progress_range
        )
        if len(page) < 1000:
            break
    return result


def download_funding(
    transport: BinanceTransport,
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    observer: DatasetBuildObserver | None = None,
    progress_range: tuple[float, float] = (0.39, 0.47),
) -> list[dict[str, Any]]:
    result = []
    cursor = start_ms
    page_count = 0
    while cursor <= end_ms:
        page = transport.get(
            "perpetual",
            endpoint("perpetual", "fundingRate"),
            {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not isinstance(page, list):
            raise DatasetValidationError(f"Invalid Binance funding response for {symbol}")
        if not page:
            break
        result.extend(page)
        next_cursor = int(page[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            raise DatasetValidationError("Binance funding pagination did not advance")
        cursor = next_cursor
        page_count += 1
        publish_download_progress(
            observer, symbol, "funding", page_count, cursor, start_ms, end_ms, progress_range
        )
        if len(page) < 1000:
            break
    return ordered_funding_events(result, symbol)


def download_aggregate_trades(
    transport: BinanceTransport,
    *,
    product: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    from_id: int | None = None
    while True:
        params: dict[str, str | int] = {"symbol": symbol, "limit": 1000}
        if from_id is None:
            params.update({"startTime": start_ms, "endTime": end_ms})
        else:
            params["fromId"] = from_id
        page = transport.get(product, endpoint(product, "aggTrades"), params)
        if not isinstance(page, list) or not page:
            break
        in_range = [item for item in page if start_ms <= int(item["T"]) <= end_ms]
        result.extend(in_range)
        last_id = int(page[-1]["a"])
        next_id = last_id + 1
        if from_id is not None and next_id <= from_id:
            raise DatasetValidationError("Binance aggregate-trade pagination did not advance")
        if len(page) < 1000 or int(page[-1]["T"]) > end_ms:
            break
        from_id = next_id
    deduplicated = {int(item["a"]): item for item in result}
    return [deduplicated[key] for key in sorted(deduplicated)]


def download_mark_klines(
    transport: BinanceTransport,
    *,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    observer: DatasetBuildObserver | None = None,
    progress_range: tuple[float, float] = (0.48, 0.62),
) -> list[list[Any]]:
    result: list[list[Any]] = []
    cursor = start_ms
    page_count = 0
    while cursor <= end_ms:
        page = transport.get(
            "perpetual",
            endpoint("perpetual", "markPriceKlines"),
            {
                "symbol": symbol,
                "interval": request_timeframe(timeframe),
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
        )
        if not isinstance(page, list) or not page:
            break
        result.extend(page)
        next_cursor = int(page[-1][6]) + 1
        if next_cursor <= cursor:
            raise DatasetValidationError("Binance mark-price pagination did not advance")
        cursor = next_cursor
        page_count += 1
        publish_download_progress(
            observer, symbol, "mark candles", page_count, cursor, start_ms, end_ms, progress_range
        )
        if len(page) < 1500:
            break
    return result


def publish_download_progress(
    observer: DatasetBuildObserver | None,
    symbol: str,
    phase: str,
    page_count: int,
    cursor: int,
    start_ms: int,
    end_ms: int,
    progress_range: tuple[float, float],
) -> None:
    if observer is None:
        return
    fraction = min(1.0, max(0.0, (cursor - start_ms) / max(1, end_ms - start_ms)))
    pct_from, pct_to = progress_range
    observer.progress(
        stage="dataset_download",
        message=f"downloading Binance {symbol} {phase}: page {page_count} ({fraction:.0%})",
        pct=pct_from + fraction * (pct_to - pct_from),
    )
