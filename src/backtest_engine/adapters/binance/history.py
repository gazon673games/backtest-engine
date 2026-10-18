"""Probe finalized candles before a download, including perpetual mark-price coverage."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, time, timedelta
from typing import Any

from backtest_engine.adapters.binance.download import canonical_timeframe, request_timeframe
from backtest_engine.adapters.binance.transport import BinanceTransport, endpoint
from backtest_engine.adapters.models import InstrumentHistoryRange
from backtest_engine.core.exceptions import DatasetValidationError


def binance_history_range(
    transport: BinanceTransport,
    *,
    engine: str,
    symbol: str,
    timeframe: str,
    now: datetime | None = None,
) -> InstrumentHistoryRange:
    if engine not in {"spot", "perpetual"}:
        raise DatasetValidationError(f"Unsupported Binance product: {engine}")
    symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,40}", symbol):
        raise DatasetValidationError("Choose a valid Binance pair")
    timeframe = canonical_timeframe(timeframe)
    retrieved_at = now or datetime.now(UTC)
    now_ms = int(retrieved_at.timestamp() * 1000)
    names = ["klines"] + (["markPriceKlines"] if engine == "perpetual" else [])
    bounds = {}
    # The four small boundary probes are independent; do not add their network latencies.
    with ThreadPoolExecutor(max_workers=2 * len(names)) as pool:
        probes = {
            name: (
                pool.submit(
                    transport.get,
                    engine,
                    endpoint(engine, name),
                    {"symbol": symbol, "interval": "1m", "startTime": 0, "limit": 1},
                ),
                pool.submit(
                    transport.get,
                    engine,
                    endpoint(engine, name),
                    {"symbol": symbol, "interval": request_timeframe(timeframe), "limit": 2},
                ),
            )
            for name in names
        }
        for name, (first, recent) in probes.items():
            first_ms = finalized_open_times(first.result(), now_ms)
            last_ms = finalized_open_times(recent.result(), now_ms)
            if not first_ms or not last_ms:
                raise DatasetValidationError(
                    f"No finalized Binance {engine} {symbol} {timeframe} {name} "
                    "history is available"
                )
            bounds[name] = (min(first_ms), max(last_ms))
    first_ms = max(value[0] for value in bounds.values())
    last_ms = min(value[1] for value in bounds.values())
    first_bar = datetime.fromtimestamp(first_ms / 1000, UTC)
    last_bar = datetime.fromtimestamp(last_ms / 1000, UTC)
    # Probe the first minute, since a first daily candle can include a partial listing day.
    # Date-only requests begin at midnight. Avoid a partial first day with missing marks.
    from_date = first_bar.date() + timedelta(days=first_bar.time() != time.min)
    till_date = last_bar.date()
    if from_date > till_date:
        raise DatasetValidationError(
            f"Binance {engine} {symbol} {timeframe} has no complete start day yet; "
            f"first finalized candle: {first_bar.isoformat()}"
        )
    return InstrumentHistoryRange(
        adapter_id="binance",
        engine=engine,
        symbol=symbol,
        timeframe=timeframe,
        first_bar=first_bar,
        last_bar=last_bar,
        from_date=from_date,
        till_date=till_date,
        retrieved_at=retrieved_at,
        components={
            name: {
                "first_bar": datetime.fromtimestamp(first / 1000, UTC).isoformat(),
                "last_bar": datetime.fromtimestamp(last / 1000, UTC).isoformat(),
            }
            for name, (first, last) in bounds.items()
        },
    )


def finalized_open_times(rows: Any, now_ms: int) -> list[int]:
    if not isinstance(rows, list):
        raise DatasetValidationError("Invalid Binance historical availability response")
    result = []
    for row in rows:
        try:
            opened, closed = int(row[0]), int(row[6])
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise DatasetValidationError("Invalid Binance historical availability candle") from exc
        if opened < 0 or closed < opened:
            raise DatasetValidationError("Invalid Binance historical availability timestamps")
        if closed < now_ms:
            result.append(opened)
    return result
