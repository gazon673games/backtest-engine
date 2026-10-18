"""Routes synchronized portfolio events through a point-in-time universe."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from backtest_engine.data.universe import PointInTimeUniverse
from backtest_engine.portfolio.multi.config import MultiInstrumentUniverseConfig


@dataclass(frozen=True)
class UniverseRoute:
    strategy_assets: list[str]
    execution_assets: list[str]
    universe_assets: tuple[str, ...]
    added: tuple[str, ...]
    removed: tuple[str, ...]
    forced_decisions: dict[str, dict[str, object]]


class MultiUniverseRuntime:
    def __init__(self, config: MultiInstrumentUniverseConfig, asset_ids: list[str]) -> None:
        self.config = config
        self.asset_ids = tuple(asset_ids)
        self.previous: set[str] = set()
        self.selector = (
            PointInTimeUniverse.from_path(
                config.path,
                ranking_column=config.ranking_column,
                require_point_in_time=config.require_point_in_time,
            )
            if config.path is not None
            else None
        )

    def route(
        self,
        current_ts: datetime,
        exact_assets: list[str],
        decision_assets: list[str],
        positions: dict[str, float],
    ) -> UniverseRoute:
        selected = self._selected(current_ts)
        added_assets = selected - self.previous
        removed_assets = self.previous - selected
        added = tuple(asset_id for asset_id in self.asset_ids if asset_id in added_assets)
        removed = tuple(asset_id for asset_id in self.asset_ids if asset_id in removed_assets)
        self.previous = selected
        strategy_assets = [asset_id for asset_id in decision_assets if asset_id in selected]
        held_assets = {asset_id for asset_id in exact_assets if abs(positions[asset_id]) > 1e-12}
        held_removed = [
            asset_id
            for asset_id in exact_assets
            if asset_id not in selected and asset_id in held_assets
        ]
        executable = set(strategy_assets) | held_assets
        execution_assets = [asset_id for asset_id in self.asset_ids if asset_id in executable]
        forced = {}
        if self.config.removal_policy == "liquidate":
            forced = {
                asset_id: {
                    "target": 0.0,
                    "reason": "universe_removed",
                    "features": {"universe_action": "liquidate"},
                }
                for asset_id in held_removed
            }
        return UniverseRoute(
            strategy_assets,
            execution_assets,
            tuple(asset_id for asset_id in self.asset_ids if asset_id in selected),
            added,
            removed,
            forced,
        )

    def _selected(self, current_ts: datetime) -> set[str]:
        if self.selector is None:
            return set(self.asset_ids)
        selection = self.selector.select(
            current_ts,
            top_n=self.config.top_n,
            min_score=self.config.min_score,
            min_price=self.config.min_price,
            allowlist=set(self.config.allowlist),
            blocklist=set(self.config.blocklist),
        )
        return set(selection.asset_ids) & set(self.asset_ids)
