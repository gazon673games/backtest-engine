from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.portfolio.multi.utils import finite_float


class MultiInstrumentStrategyContext:
    """No-lookahead context for portfolio strategies over several intraday datasets."""

    def __init__(
        self,
        *,
        current_ts: datetime,
        frames: dict[str, pl.DataFrame],
        visible_indices: dict[str, int],
        tradable_assets: list[str],
        previous_targets: dict[str, float],
        previous_contracts: dict[str, float],
        equity_cash: float,
        universe_assets: tuple[str, ...] | None = None,
        universe_added: tuple[str, ...] = (),
        universe_removed: tuple[str, ...] = (),
    ) -> None:
        self.current_ts = current_ts
        self.tradable_assets = tuple(tradable_assets)
        self.asset_ids = tuple(frames)
        self.previous_targets = dict(previous_targets)
        self.previous_contracts = dict(previous_contracts)
        self.equity_cash = equity_cash
        self.universe_assets = universe_assets or tuple(frames)
        self.universe_added = universe_added
        self.universe_removed = universe_removed
        self.state: dict[str, Any] = {}
        self._frames = frames
        self._visible_indices = visible_indices

    def history(self, asset_id: str) -> pl.DataFrame:
        index = self._visible_index(asset_id)
        if index < 0:
            return self._frames[asset_id].head(0)
        return self._frames[asset_id].slice(0, index + 1)

    def current_bar(self, asset_id: str) -> dict[str, Any] | None:
        index = self._visible_index(asset_id)
        if index < 0:
            return None
        return self._frames[asset_id].row(index, named=True)

    def series(self, asset_id: str, column: str) -> pl.Series:
        history = self.history(asset_id)
        if column not in history.columns:
            raise StrategyPluginError(f"Missing column for {asset_id}: {column}")
        return history.get_column(column)

    def latest_close(self, asset_id: str) -> float | None:
        row = self.current_bar(asset_id)
        if row is None:
            return None
        return finite_float(row["close"], f"{asset_id}.close")

    def sma(self, asset_id: str, column: str, window: int) -> float | None:
        if window <= 0:
            raise StrategyPluginError("sma window must be greater than 0")
        values = self.series(asset_id, column)
        if len(values) < window:
            return None
        tail = values.tail(window)
        if tail.null_count() > 0:
            return None
        return float(tail.mean())

    def _visible_index(self, asset_id: str) -> int:
        if asset_id not in self._frames:
            raise StrategyPluginError(f"Unknown asset_id: {asset_id}")
        return self._visible_indices.get(asset_id, -1)
