"""Normalizes historical Binance-style L2 snapshot and delta archives."""

from __future__ import annotations

from itertools import groupby
from typing import Any

import polars as pl

from backtest_engine.adapters.binance.normalize import as_datetime, tick_base_row
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.asset_schema import normalize_intraday_asset_frame
from backtest_engine.execution.book_state import L2OrderBook

REQUIRED_BOOK_COLUMNS = frozenset({"ts", "sequence", "update_type", "side", "price", "size"})


def normalize_order_book_frame(
    *,
    product: str,
    symbol: str,
    metadata: dict[str, Any],
    frame: pl.DataFrame,
) -> pl.DataFrame:
    missing = sorted(REQUIRED_BOOK_COLUMNS.difference(frame.columns))
    if missing:
        raise DatasetValidationError("Order-book archive is missing columns: " + ", ".join(missing))
    source_rows = [normalize_source_row(row) for row in frame.iter_rows(named=True)]
    source_rows.sort(key=lambda row: (row["ts"], row["sequence"]))
    book = L2OrderBook()
    output: list[dict[str, Any]] = []
    for _, grouped in groupby(source_rows, key=book_batch_key):
        batch = list(grouped)
        book.apply_batch(batch)
        bid, ask = book.best_bid(), book.best_ask()
        if bid is None or ask is None:
            raise DatasetValidationError(
                "Every L2 batch must leave both bid and ask sides populated"
            )
        midpoint = (bid + ask) / 2.0
        for index, row in enumerate(batch):
            default_id = f"{row['sequence']}:{row['side']}:{row['price']}:{index}"
            event_id = row.get("event_id") or default_id
            output.append(
                tick_base_row(product, symbol, metadata, row["ts"], midpoint)
                | {
                    **row,
                    "event_id": str(event_id),
                    "bid": bid,
                    "ask": ask,
                    "volume": float(row["size"]),
                    "value": 0.0,
                    "data_kind": "book",
                }
            )
    if not output:
        raise DatasetValidationError("Order-book archive is empty")
    return normalize_intraday_asset_frame(pl.DataFrame(output, orient="row"))


def normalize_source_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["ts"] = as_datetime(row["ts"])
    result["sequence"] = int(row["sequence"])
    result["previous_sequence"] = (
        None if row.get("previous_sequence") is None else int(row["previous_sequence"])
    )
    result["update_type"] = str(row["update_type"]).strip().lower()
    result["side"] = str(row["side"]).strip().lower()
    result["price"] = float(row["price"])
    result["size"] = float(row["size"])
    return result


def book_batch_key(row: dict[str, Any]) -> tuple[Any, int]:
    return row["ts"], int(row["sequence"])
