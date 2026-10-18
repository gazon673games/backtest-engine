"""Prepare visible strategy context and requested multi-leg contract plans."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig
from backtest_engine.portfolio.multi.data import instrument_spec_for_row
from backtest_engine.portfolio.multi.risk import (
    target_contracts as target_contracts_for_signal,
)


def visible_indices_at(
    timestamps: dict[str, list[datetime]],
    current_ts: datetime,
) -> dict[str, int]:
    return {
        asset_id: bisect_right(asset_ts, current_ts) - 1
        for asset_id, asset_ts in timestamps.items()
    }


def requested_contract_plan(
    *,
    current_ts: datetime,
    tradable_assets: list[str],
    decisions: dict[str, dict[str, Any]],
    exact_indices: dict[str, dict[datetime, int]],
    rows_by_asset: dict[str, Sequence[dict[str, Any]]],
    loaded_legs: dict[str, dict[str, Any]],
    config: MultiInstrumentBacktestConfig,
    current_targets: dict[str, float],
    current_contracts: dict[str, float],
    equity_cash: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, str | None], dict[str, dict[str, Any]]]:
    requested_targets = dict(current_targets)
    requested_contracts = dict(current_contracts)
    reasons: dict[str, str | None] = {}
    features: dict[str, dict[str, Any]] = {}
    for asset_id in tradable_assets:
        decision = decisions.get(asset_id)
        index = exact_indices[asset_id][current_ts]
        row = rows_by_asset[asset_id][index]
        if decision is None:
            continue
        requested_targets[asset_id] = decision["target"]
        reasons[asset_id] = decision.get("reason")
        features[asset_id] = decision.get("features", {})
        if "contracts" in decision:
            requested_contracts[asset_id] = decision["contracts"] * decision["target"]
            continue
        requested_contracts[asset_id] = target_contracts_for_signal(
            requested_targets[asset_id],
            float(row["close"]),
            instrument_spec_for_row(row, current_ts, loaded_legs[asset_id]["config"]),
            config.portfolio,
            equity_cash,
        )
    return requested_targets, requested_contracts, reasons, features


def scaled_targets(
    tradable_assets: list[str],
    requested_targets: dict[str, float],
    requested_contracts: dict[str, float],
    target_contracts: dict[str, float],
    current_targets: dict[str, float],
    current_contracts: dict[str, float],
) -> dict[str, float]:
    targets = dict(requested_targets)
    for asset_id in tradable_assets:
        if target_contracts[asset_id] == current_contracts[asset_id]:
            targets[asset_id] = current_targets[asset_id]
        elif requested_contracts[asset_id] != 0.0:
            targets[asset_id] = requested_targets[asset_id] * (
                target_contracts[asset_id] / requested_contracts[asset_id]
            )
    return targets
