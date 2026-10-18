from __future__ import annotations

from typing import Any

import polars as pl

from backtest_engine.accounting.futures import (
    position_notional as futures_position_notional,
)
from backtest_engine.accounting.ledger_runtime import (
    amount_in_base_currency,
)
from backtest_engine.portfolio.multi.config import (
    MultiInstrumentInstrumentSpec,
    MultiInstrumentPortfolioConfig,
)
from backtest_engine.portfolio.multi.data import instrument_spec_for_row
from backtest_engine.portfolio.multi.futures import (
    futures_instrument_spec,
    margin_requirement,
)
from backtest_engine.portfolio.multi.utils import as_datetime, finite_float


def apply_position_notional_limit(
    *,
    requested_contracts: dict[str, float],
    previous_contracts: dict[str, float],
    tradable_assets: list[str],
    frames: dict[str, pl.DataFrame],
    visible_indices: dict[str, int],
    loaded_legs: dict[str, dict[str, Any]],
    portfolio: MultiInstrumentPortfolioConfig,
) -> dict[str, float]:
    if portfolio.max_position_notional is None:
        return dict(requested_contracts)
    target = dict(requested_contracts)
    for asset_id in tradable_assets:
        maybe_clip_asset_position(
            asset_id,
            target,
            previous_contracts,
            frames,
            visible_indices,
            loaded_legs,
            portfolio,
        )
    return target


def maybe_clip_asset_position(
    asset_id: str,
    target: dict[str, float],
    previous_contracts: dict[str, float],
    frames: dict[str, pl.DataFrame],
    visible_indices: dict[str, int],
    loaded_legs: dict[str, dict[str, Any]],
    portfolio: MultiInstrumentPortfolioConfig,
) -> None:
    index = visible_indices.get(asset_id, -1)
    if index < 0:
        return
    row = loaded_legs[asset_id]["rows"][index]
    instrument = instrument_spec_for_row(
        row,
        as_datetime(row["ts"]),
        loaded_legs[asset_id]["config"],
    )
    notional = futures_position_notional(
        target[asset_id],
        close_at(frames[asset_id], index),
        futures_instrument_spec(instrument),
    )
    notional = instrument_cash_in_base(notional, instrument, portfolio)
    if notional <= portfolio.max_position_notional:
        return
    if portfolio.risk_policy == "reject":
        target[asset_id] = previous_contracts[asset_id]
        return
    scale = 0.0 if notional == 0.0 else portfolio.max_position_notional / notional
    target[asset_id] *= scale


def gross_notional(
    *,
    contracts: dict[str, float],
    frames: dict[str, pl.DataFrame],
    visible_indices: dict[str, int],
    loaded_legs: dict[str, dict[str, Any]],
    portfolio: MultiInstrumentPortfolioConfig,
) -> float:
    total = 0.0
    for asset_id, contract_count in contracts.items():
        if contract_count == 0.0:
            continue
        index = visible_indices.get(asset_id, -1)
        if index < 0:
            continue
        row = loaded_legs[asset_id]["rows"][index]
        instrument = instrument_spec_for_row(
            row,
            as_datetime(row["ts"]),
            loaded_legs[asset_id]["config"],
        )
        total += instrument_cash_in_base(
            futures_position_notional(
                contract_count,
                close_at(frames[asset_id], index),
                futures_instrument_spec(instrument),
            ),
            instrument,
            portfolio,
        )
    return total


def portfolio_margin_requirement(
    *,
    contracts: dict[str, float],
    frames: dict[str, pl.DataFrame],
    visible_indices: dict[str, int],
    loaded_legs: dict[str, dict[str, Any]],
    portfolio: MultiInstrumentPortfolioConfig,
) -> float:
    del frames
    total = 0.0
    for asset_id, contract_count in contracts.items():
        if contract_count == 0.0:
            continue
        index = visible_indices.get(asset_id, -1)
        if index < 0:
            continue
        row = loaded_legs[asset_id]["rows"][index]
        instrument = instrument_spec_for_row(
            row,
            as_datetime(row["ts"]),
            loaded_legs[asset_id]["config"],
        )
        total += instrument_cash_in_base(
            margin_requirement(contract_count, instrument),
            instrument,
            portfolio,
        )
    return total


def contracts_scale(
    requested_contracts: dict[str, float],
    target: dict[str, float],
    tradable_assets: list[str],
) -> float:
    ratios = [
        abs(target[asset_id] / requested_contracts[asset_id])
        for asset_id in tradable_assets
        if requested_contracts[asset_id] != 0.0
    ]
    return min(ratios) if ratios else 1.0


def close_at(frame: pl.DataFrame, index: int) -> float:
    return finite_float(frame.get_column("close")[index], "close")


def instrument_cash_in_base(
    amount: float,
    instrument: MultiInstrumentInstrumentSpec,
    portfolio: MultiInstrumentPortfolioConfig,
) -> float:
    return amount_in_base_currency(
        amount,
        instrument.pnl_currency,
        portfolio.base_currency,
        portfolio.fx_rates,
    )
