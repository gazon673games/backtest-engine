"""Revalue funding in an immutable local prefix without downloading trade candles."""

from bisect import bisect_left, bisect_right
from math import isclose
from typing import Any

import polars as pl

from backtest_engine.adapters.binance.download import download_funding
from backtest_engine.adapters.binance.funding import funding_price_rate
from backtest_engine.adapters.binance.funding_recovery import recover_funding_prices
from backtest_engine.adapters.binance.transport import BinanceTransport
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.timeframes import parse_timeframe_minutes


def recover_prefix_funding(
    frame: pl.DataFrame,
    transport: BinanceTransport,
    symbol: str,
    sources: list[dict[str, Any]],
) -> pl.DataFrame:
    if not frame.filter(pl.col("funding_event") & pl.col("funding_price_rate").is_null()).height:
        return frame
    opened = [int(stamp.timestamp() * 1000) for stamp in frame["ts"]]
    step = parse_timeframe_minutes(frame["timeframe"][0]) * 60_000
    # Aggregated bars can use an exclusive close timestamp or delayed availability.
    # Neither makes a funding event at the next bucket's open part of this bucket.
    closed = [
        min(int(stamp.timestamp() * 1000), first + step - 1)
        for first, stamp in zip(opened, frame["bar_close_ts"], strict=True)
    ]
    events = download_funding(transport, symbol=symbol, start_ms=opened[0], end_ms=closed[-1])
    marks = [[stamp, price] for stamp, price in zip(opened, frame["mark_open"], strict=True)]
    events = recover_funding_prices(
        transport, symbol=symbol, events=events, marks=marks, sources=sources
    )
    events.sort(key=lambda event: int(event["fundingTime"]))
    times = [int(event["fundingTime"]) for event in events]
    prices, estimated = [], []
    for index, (first, last) in enumerate(zip(opened, closed, strict=True)):
        selected = events[bisect_left(times, first) : bisect_right(times, last)]
        rate = sum(float(event["fundingRate"]) for event in selected)
        if bool(selected) != frame["funding_event"][index] or not isclose(
            rate, frame["funding_rate"][index], rel_tol=1e-10, abs_tol=1e-12
        ):
            raise DatasetValidationError(
                f"Binance funding history changed for {symbol} at {frame['ts'][index]}; "
                "prefix recovery cannot replace event rates or timestamps"
            )
        old = frame["funding_price_rate"][index]
        price = funding_price_rate(selected, symbol)
        if old is not None:
            prices.append(old)
            estimated.append(frame["funding_price_estimated"][index])
        else:
            prices.append(price)
            estimated.append(any(event.get("fundingPriceEstimated", False) for event in selected))
    return frame.with_columns(
        pl.Series("funding_price_rate", prices, dtype=pl.Float64),
        pl.Series("funding_price_estimated", estimated, dtype=pl.Boolean),
    )
