"""Defines the exchange-neutral discovery and dataset-build adapter protocol."""

from __future__ import annotations

from typing import Protocol

from backtest_engine.adapters.build_events import DatasetBuildObserver
from backtest_engine.adapters.models import (
    AdapterInfo,
    AdapterItem,
    DatasetBuildRequest,
    DatasetBuildResult,
)


class MarketDataAdapter(Protocol):
    id: str

    def info(self) -> AdapterInfo: ...

    def engines(self) -> list[AdapterItem]: ...

    def markets(self, engine: str) -> list[AdapterItem]: ...

    def boards(self, engine: str, market: str) -> list[AdapterItem]: ...

    def instruments(
        self,
        *,
        engine: str,
        market: str,
        board: str,
        query: str,
    ) -> list[dict[str, object]]: ...

    def build_dataset(
        self,
        request: DatasetBuildRequest,
        observer: DatasetBuildObserver | None = None,
    ) -> DatasetBuildResult: ...
