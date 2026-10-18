"""Public closed-candle Binance feed implementing the paper event boundary."""

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from backtest_engine.adapters.binance.transport import (
    BinanceHttpTransport,
    BinanceTransport,
    endpoint,
)
from backtest_engine.core.exceptions import DatasetValidationError


class BinanceClosedKlineFeed:
    """Polls the most recent closed public Binance candle without credentials."""

    def __init__(
        self,
        *,
        symbol: str,
        timeframe: str,
        product: str = "spot",
        transport: BinanceTransport | None = None,
        poll_seconds: float = 1.0,
    ) -> None:
        if product not in {"spot", "perpetual"}:
            raise DatasetValidationError("Binance paper product must be spot or perpetual")
        if poll_seconds <= 0.0:
            raise DatasetValidationError("Binance paper poll_seconds must be positive")
        self.symbol = symbol.strip().upper()
        self.timeframe = timeframe
        self.product = product
        self.transport = transport or BinanceHttpTransport()
        self.poll_seconds = poll_seconds
        self._last_open_ms: int | None = None

    def poll(self, now: datetime | None = None) -> dict[str, Any] | None:
        payload = self.transport.get(
            self.product,
            endpoint(self.product, "klines"),
            {"symbol": self.symbol, "interval": self.timeframe, "limit": 2},
        )
        if not isinstance(payload, list):
            raise DatasetValidationError("Binance kline response must be a list")
        current_ms = int((now or datetime.now(UTC)).timestamp() * 1_000)
        closed = [item for item in payload if len(item) >= 7 and int(item[6]) < current_ms]
        if not closed:
            return None
        item = closed[-1]
        open_ms = int(item[0])
        if self._last_open_ms is not None and open_ms <= self._last_open_ms:
            return None
        self._last_open_ms = open_ms
        return {
            "ts": _from_ms(open_ms),
            "available_at": _from_ms(int(item[6]) + 1),
            "timeframe": self.timeframe,
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
            "symbol": self.symbol,
        }

    def events(self, *, max_events: int | None = None) -> Iterator[dict[str, Any]]:
        emitted = 0
        while max_events is None or emitted < max_events:
            row = self.poll()
            if row is not None:
                emitted += 1
                yield row
            if max_events is None or emitted < max_events:
                time.sleep(self.poll_seconds)


def _from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1_000.0, tz=UTC)
