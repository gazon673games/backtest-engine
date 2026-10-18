"""Streaming event feeds for paper trading and deterministic replay."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from backtest_engine.core.exceptions import DatasetValidationError


class ParquetPaperFeed:
    def __init__(self, path: Path, *, batch_size: int = 8_192) -> None:
        if not path.exists():
            raise DatasetValidationError(f"Paper feed does not exist: {path}")
        if batch_size <= 0:
            raise DatasetValidationError("Paper feed batch_size must be positive")
        self.path = path
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        source = pq.ParquetFile(self.path)
        for batch in source.iter_batches(batch_size=self.batch_size):
            yield from batch.to_pylist()


def __getattr__(name: str) -> Any:
    # Legacy public import. Normal replay/engine imports never load a provider.
    if name == "BinanceClosedKlineFeed":
        from backtest_engine.adapters.binance.paper_feed import BinanceClosedKlineFeed

        return BinanceClosedKlineFeed
    raise AttributeError(name)
