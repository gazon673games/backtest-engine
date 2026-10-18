"""Normalizes Binance symbols, klines, and funding to canonical event rows."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import UTC, datetime
from typing import Any

import polars as pl

from backtest_engine.adapters.binance.funding import funding_price_rate, ordered_funding_events
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.asset_schema import normalize_intraday_asset_frame


def symbol_metadata(exchange_info: dict[str, Any], symbol: str) -> dict[str, Any]:
    symbols = exchange_info.get("symbols", [])
    match = next((item for item in symbols if item.get("symbol") == symbol), None)
    if match is None:
        raise DatasetValidationError(f"Binance symbol not found: {symbol}")
    return match


def normalize_klines(
    *,
    product: str,
    symbol: str,
    timeframe: str,
    metadata: dict[str, Any],
    klines: list[list[Any]],
    funding: list[dict[str, Any]] | None = None,
    mark_klines: list[list[Any]] | None = None,
) -> pl.DataFrame:
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    funding_rows = ordered_funding_events(funding or [], symbol)
    funding_times = [int(item["fundingTime"]) for item in funding_rows]
    marks = {int(item[0]): item for item in mark_klines or []}
    rows = []
    for raw in klines:
        open_ms = int(raw[0])
        close_ms = int(raw[6])
        if close_ms > now_ms:
            continue
        events = funding_rows[
            bisect_left(funding_times, open_ms) : bisect_right(funding_times, close_ms)
        ]
        rows.append(
            normalized_row(
                product=product,
                symbol=symbol,
                timeframe=timeframe,
                metadata=metadata,
                raw=raw,
                funding_rate=sum((float(item["fundingRate"]) for item in events), 0.0),
                mark=marks.get(open_ms),
            )
        )
        rows[-1]["funding_event"] = bool(events)
        if product == "perpetual":
            rows[-1]["funding_price_rate"] = funding_price_rate(events, symbol)
            rows[-1]["funding_price_estimated"] = any(
                item.get("fundingPriceEstimated", False) for item in events
            )
    if not rows:
        raise DatasetValidationError(f"Binance returned no finalized klines for {symbol}")
    # A chunk can start with many bars without funding or mark observations.
    # Explicit numeric types keep independently normalized archive chunks compatible.
    numeric = {"funding_rate": pl.Float64}
    if product == "perpetual":
        numeric.update(
            dict.fromkeys(
                ("funding_price_rate", "mark_open", "mark_high", "mark_low", "mark_price"),
                pl.Float64,
            )
        )
    frame = pl.DataFrame(rows, orient="row", schema_overrides=numeric)
    return normalize_intraday_asset_frame(frame)


def normalized_row(
    *,
    product: str,
    symbol: str,
    timeframe: str,
    metadata: dict[str, Any],
    raw: list[Any],
    funding_rate: float,
    mark: list[Any] | None,
) -> dict[str, Any]:
    base_currency = str(metadata["baseAsset"])
    quote_currency = str(metadata["quoteAsset"])
    settlement = str(metadata.get("marginAsset") or quote_currency)
    open_ts = datetime.fromtimestamp(int(raw[0]) / 1000.0, tz=UTC)
    close_ts = datetime.fromtimestamp(int(raw[6]) / 1000.0, tz=UTC)
    row = {
        "source": "binance_rest",
        "exchange": "BINANCE",
        "engine": product,
        "market": "crypto",
        "board": "SPOT" if product == "spot" else "USD_M_PERPETUAL",
        "symbol": symbol,
        "timeframe": timeframe,
        "ts": open_ts,
        "bar_open_ts": open_ts,
        "bar_close_ts": close_ts,
        "available_at": close_ts,
        "is_final": True,
        "open": float(raw[1]),
        "high": float(raw[2]),
        "low": float(raw[3]),
        "close": float(raw[4]),
        "volume": float(raw[5]),
        "value": float(raw[7]),
        "currency": quote_currency,
        "dataset_type": "spot" if product == "spot" else "perpetual",
        "asset_class": "crypto",
        "instrument_type": "spot" if product == "spot" else "perpetual",
        "base_currency": base_currency,
        "quote_currency": quote_currency,
        "settlement_currency": settlement,
        "funding_rate": funding_rate,
        "funding_event": funding_rate != 0.0,
        "data_kind": "bars",
    }
    if product == "perpetual":
        row.update(
            {
                "mark_open": float(mark[1]) if mark else None,
                "mark_high": float(mark[2]) if mark else None,
                "mark_low": float(mark[3]) if mark else None,
                "mark_price": float(mark[4]) if mark else None,
            }
        )
    return row


def normalize_aggregate_trades(
    *,
    product: str,
    symbol: str,
    metadata: dict[str, Any],
    trades: list[dict[str, Any]],
) -> pl.DataFrame:
    rows = []
    for item in trades:
        ts = timestamp_value(int(item["T"]))
        price = float(item["p"])
        quantity = float(item["q"])
        rows.append(
            tick_base_row(product, symbol, metadata, ts, price)
            | {
                "event_id": str(item["a"]),
                "trade_price": price,
                "trade_size": quantity,
                "volume": quantity,
                "value": price * quantity,
                "aggressor_side": "sell" if bool(item["m"]) else "buy",
                "data_kind": "trades",
            }
        )
    if not rows:
        raise DatasetValidationError(f"Binance returned no aggregate trades for {symbol}")
    return normalize_intraday_asset_frame(pl.DataFrame(rows, orient="row").sort(["ts", "event_id"]))


def normalize_quote_frame(
    *,
    product: str,
    symbol: str,
    metadata: dict[str, Any],
    frame: pl.DataFrame,
) -> pl.DataFrame:
    required = {"ts", "bid", "ask"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DatasetValidationError("Quote archive is missing columns: " + ", ".join(missing))
    rows = []
    for index, item in enumerate(frame.sort("ts").iter_rows(named=True)):
        bid = float(item["bid"])
        ask = float(item["ask"])
        if bid <= 0.0 or ask <= 0.0 or bid > ask:
            raise DatasetValidationError(f"Invalid quote at row {index}: bid={bid}, ask={ask}")
        midpoint = (bid + ask) / 2.0
        ts = as_datetime(item["ts"])
        bid_size = optional_float(item.get("bid_size"))
        ask_size = optional_float(item.get("ask_size"))
        quote_size = (
            min(value for value in (bid_size, ask_size) if value is not None)
            if (bid_size is not None or ask_size is not None)
            else 0.0
        )
        rows.append(
            tick_base_row(product, symbol, metadata, ts, midpoint)
            | {
                "event_id": str(item.get("event_id") or index),
                "bid": bid,
                "ask": ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "quote_size": quote_size,
                "volume": quote_size,
                "value": 0.0,
                "data_kind": "quotes",
            }
        )
    if not rows:
        raise DatasetValidationError("Quote archive is empty")
    return normalize_intraday_asset_frame(pl.DataFrame(rows, orient="row"))


def tick_base_row(
    product: str,
    symbol: str,
    metadata: dict[str, Any],
    ts: datetime,
    price: float,
) -> dict[str, Any]:
    base = str(metadata["baseAsset"])
    quote = str(metadata["quoteAsset"])
    settlement = str(metadata.get("marginAsset") or quote)
    return {
        "source": "binance_archive",
        "exchange": "BINANCE",
        "engine": product,
        "market": "crypto",
        "board": "SPOT" if product == "spot" else "USD_M_PERPETUAL",
        "symbol": symbol,
        "timeframe": "1m",
        "ts": ts,
        "bar_open_ts": ts,
        "bar_close_ts": ts,
        "available_at": ts,
        "is_final": True,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "currency": quote,
        "dataset_type": "spot" if product == "spot" else "perpetual",
        "asset_class": "crypto",
        "instrument_type": "spot" if product == "spot" else "perpetual",
        "base_currency": base,
        "quote_currency": quote,
        "settlement_currency": settlement,
        "funding_rate": 0.0,
        "funding_event": False,
    }


def timestamp_value(raw: int) -> datetime:
    divisor = 1_000_000.0 if raw >= 10**15 else 1_000.0
    return datetime.fromtimestamp(raw / divisor, tz=UTC)


def as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    raise DatasetValidationError(f"Quote timestamp must be datetime, got {value!r}")


def optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def instrument_payload(product: str, metadata: dict[str, Any]) -> dict[str, Any]:
    filters = {item["filterType"]: item for item in metadata.get("filters", [])}
    lot = filters.get("LOT_SIZE", {})
    price = filters.get("PRICE_FILTER", {})
    minimum_filter = filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {}))
    minimum_notional = minimum_filter.get("minNotional", minimum_filter.get("notional"))
    quote = str(metadata["quoteAsset"])
    return {
        "symbol": str(metadata["symbol"]),
        "asset_class": "crypto",
        "instrument_type": "spot" if product == "spot" else "perpetual",
        "contract_type": "spot" if product == "spot" else "linear",
        "price_currency": quote,
        "pnl_currency": str(metadata.get("marginAsset") or quote),
        "base_currency": str(metadata["baseAsset"]),
        "quote_currency": quote,
        "settlement_currency": str(metadata.get("marginAsset") or quote),
        "tick_size": optional_positive(price.get("tickSize")),
        "quantity_step": optional_positive(lot.get("stepSize")),
        "minimum_quantity": optional_positive(lot.get("minQty")),
        "minimum_notional": optional_positive(minimum_notional),
    }


def optional_positive(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if parsed > 0.0 else None
