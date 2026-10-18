from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


class FixtureTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str | int]]] = []

    def get(
        self,
        product: str,
        path: str,
        params: dict[str, str | int] | None = None,
    ) -> Any:
        self.calls.append((product, path, params or {}))
        if path.endswith("exchangeInfo"):
            return exchange_info(product)
        if path.endswith("fundingRate"):
            return [
                {
                    "symbol": "BTCUSDT",
                    "fundingTime": timestamp_ms(2024, 1, 1, 0, 0, 30),
                    "fundingRate": "0.0001",
                }
            ]
        if path.endswith("markPriceKlines"):
            mark = kline()
            mark[1:5] = ["39990", "40090", "39890", "40040"]
            return [mark]
        if path.endswith("klines"):
            return [kline()]
        if path.endswith("aggTrades"):
            return [
                {
                    "a": 1,
                    "p": "40050",
                    "q": "0.25",
                    "f": 10,
                    "l": 11,
                    "T": timestamp_ms(2024, 1, 1, 0, 0, 30),
                    "m": False,
                }
            ]
        raise AssertionError(path)


def timestamp_ms(*parts: int) -> int:
    return int(datetime(*parts, tzinfo=UTC).timestamp() * 1000)


def kline() -> list[Any]:
    return [
        timestamp_ms(2024, 1, 1),
        "40000",
        "40100",
        "39900",
        "40050",
        "10",
        timestamp_ms(2024, 1, 1, 0, 0, 59),
        "400500",
        100,
        "5",
        "200000",
        "0",
    ]


def exchange_info(product: str) -> dict[str, Any]:
    return {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "marginAsset": "USDT" if product == "perpetual" else None,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                    {
                        "filterType": "LOT_SIZE",
                        "stepSize": "0.001",
                        "minQty": "0.001",
                    },
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                ],
            }
        ]
    }
