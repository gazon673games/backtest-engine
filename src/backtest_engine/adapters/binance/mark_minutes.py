"""Recover a small mark-price outage only from complete observed minute candles."""

from decimal import Decimal, InvalidOperation
from typing import Any

from backtest_engine.adapters.binance.cache import CachedTransport
from backtest_engine.adapters.binance.download import download_mark_klines


def complete_minute_marks(
    transport: CachedTransport,
    symbol: str,
    bars: list[list[Any]],
    step: int,
    sources: list[dict[str, Any]],
) -> dict[int, list[Any]]:
    if step <= 60_000 or not bars or int(bars[-1][0]) - int(bars[0][0]) + step > 3_600_000:
        return {}
    rows = download_mark_klines(
        transport, symbol=symbol, timeframe="1m", start_ms=int(bars[0][0]), end_ms=int(bars[-1][6])
    )
    minutes = {int(row[0]): row for row in rows}
    added = {}
    if len(minutes) == len(rows):
        for bar in bars:
            start = int(bar[0])
            expected = range(start, start + step, 60_000)
            if not all(
                stamp in minutes and int(minutes[stamp][6]) == stamp + 59_999 for stamp in expected
            ):
                continue
            observed = [minutes[stamp] for stamp in expected]
            if not all(valid_minute(row) for row in observed):
                continue
            merged = list(observed[0])
            merged[2] = max(observed, key=lambda row: Decimal(row[2]))[2]
            merged[3] = min(observed, key=lambda row: Decimal(row[3]))[3]
            merged[4], merged[6] = observed[-1][4], observed[-1][6]
            added[start] = merged
    wanted = {int(bar[0]) for bar in bars}
    sources.append(
        {
            "resource": "markPriceKlines",
            "policy": "aggregate_complete_observed_1m_marks",
            "requested_timestamps": sorted(wanted),
            "recovered_timestamps": sorted(added),
            "unresolved_timestamps": sorted(wanted - added.keys()),
            "observed_minute_rows": len(rows),
        }
    )
    return added


def valid_minute(row: list[Any]) -> bool:
    try:
        opening, high, low, close = map(Decimal, row[1:5])
    except (InvalidOperation, TypeError, ValueError):
        return False
    return (
        all(value.is_finite() and value > 0 for value in (opening, high, low, close))
        and low <= min(opening, close) <= max(opening, close) <= high
    )
