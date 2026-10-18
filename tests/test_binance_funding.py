from __future__ import annotations

import pytest
from binance_fixture import exchange_info, kline

from backtest_engine.adapters.binance.normalize import normalize_klines


def test_funding_is_assigned_to_binance_open_and_close_inclusive_interval() -> None:
    first = kline()
    second = kline()
    # Binance close milliseconds precede the next open; bars never share an endpoint.
    second[0] = first[0] + 60_000
    second[6] = first[6] + 60_000
    funding = [
        {"fundingTime": second[6], "fundingRate": "0.4"},
        {"fundingTime": first[0], "fundingRate": "1.0"},
        {"fundingTime": first[6], "fundingRate": "0.2"},
        {"fundingTime": second[6] + 1, "fundingRate": "2.0"},
        {"fundingTime": first[0] + 1, "fundingRate": "0.1"},
    ]

    frame = normalize_klines(
        product="perpetual",
        symbol="BTCUSDT",
        timeframe="1m",
        metadata=exchange_info("perpetual")["symbols"][0],
        klines=[first, second],
        funding=funding,
    )

    assert frame.get_column("funding_rate").to_list() == pytest.approx([1.3, 0.4])
    assert frame.get_column("funding_event").to_list() == [True, True]
