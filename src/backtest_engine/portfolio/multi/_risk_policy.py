from __future__ import annotations

from dataclasses import replace
from typing import Any

import polars as pl

from backtest_engine.accounting.futures import (
    target_contracts as futures_target_contracts,
)
from backtest_engine.accounting.ledger_runtime import (
    amount_from_base_currency,
)
from backtest_engine.portfolio.multi._risk_math import (
    apply_position_notional_limit,
    contracts_scale,
    gross_notional,
    portfolio_margin_requirement,
)
from backtest_engine.portfolio.multi.config import (
    MultiInstrumentInstrumentSpec,
    MultiInstrumentPortfolioConfig,
)
from backtest_engine.portfolio.multi.futures import (
    futures_instrument_spec,
    futures_portfolio_spec,
)


def target_contracts(
    target: float,
    price: float,
    instrument: MultiInstrumentInstrumentSpec,
    portfolio: MultiInstrumentPortfolioConfig,
    equity_cash: float,
) -> float:
    native_portfolio = futures_portfolio_spec(portfolio)
    if native_portfolio.target_notional is not None:
        native_portfolio = replace(
            native_portfolio,
            target_notional=amount_from_base_currency(
                native_portfolio.target_notional,
                instrument.pnl_currency,
                portfolio.base_currency,
                portfolio.fx_rates,
            ),
        )
    return futures_target_contracts(
        signal=target,
        fill_price=price,
        instrument=futures_instrument_spec(instrument),
        portfolio=native_portfolio,
        equity_cash=equity_cash,
    )


def apply_portfolio_risk(
    *,
    requested_contracts: dict[str, float],
    previous_contracts: dict[str, float],
    tradable_assets: list[str],
    frames: dict[str, pl.DataFrame],
    visible_indices: dict[str, int],
    loaded_legs: dict[str, dict[str, Any]],
    equity_cash: float,
    portfolio: MultiInstrumentPortfolioConfig,
) -> dict[str, Any]:
    target = apply_position_notional_limit(
        requested_contracts=requested_contracts,
        previous_contracts=previous_contracts,
        tradable_assets=tradable_assets,
        frames=frames,
        visible_indices=visible_indices,
        loaded_legs=loaded_legs,
        portfolio=portfolio,
    )
    status = "clipped" if target != requested_contracts else "ok"
    scale = contracts_scale(requested_contracts, target, tradable_assets)
    if portfolio.max_leverage is None and portfolio.max_margin_usage is None:
        return {"target_contracts": target, "status": status, "scale": scale}
    return apply_portfolio_level_limits(
        target,
        previous_contracts,
        tradable_assets,
        frames,
        visible_indices,
        loaded_legs,
        equity_cash,
        portfolio,
        status,
        scale,
    )


def apply_portfolio_level_limits(
    target: dict[str, float],
    previous_contracts: dict[str, float],
    tradable_assets: list[str],
    frames: dict[str, pl.DataFrame],
    visible_indices: dict[str, int],
    loaded_legs: dict[str, dict[str, Any]],
    equity_cash: float,
    portfolio: MultiInstrumentPortfolioConfig,
    status: str,
    scale: float,
) -> dict[str, Any]:
    requested_notional = gross_notional(
        contracts=target,
        frames=frames,
        visible_indices=visible_indices,
        loaded_legs=loaded_legs,
        portfolio=portfolio,
    )
    requested_margin = portfolio_margin_requirement(
        contracts=target,
        frames=frames,
        visible_indices=visible_indices,
        loaded_legs=loaded_legs,
        portfolio=portfolio,
    )
    limit_scales = limit_scales_for_portfolio(
        requested_notional,
        requested_margin,
        equity_cash,
        portfolio,
    )
    if not limit_scales:
        return {"target_contracts": target, "status": status, "scale": scale}
    if portfolio.risk_policy == "reject":
        rejected = dict(target)
        for asset_id in tradable_assets:
            rejected[asset_id] = previous_contracts[asset_id]
        return {"target_contracts": rejected, "status": "rejected", "scale": 0.0}
    aggregate_scale = min(limit_scales)
    clipped = dict(target)
    for asset_id in tradable_assets:
        clipped[asset_id] = target[asset_id] * aggregate_scale
    return {"target_contracts": clipped, "status": "clipped", "scale": aggregate_scale}


def limit_scales_for_portfolio(
    requested_notional: float,
    requested_margin: float,
    equity_cash: float,
    portfolio: MultiInstrumentPortfolioConfig,
) -> list[float]:
    if equity_cash <= 0.0 and (requested_notional > 0.0 or requested_margin > 0.0):
        return [0.0]
    limit_scales: list[float] = []
    if portfolio.max_leverage is not None:
        max_notional = max(equity_cash * portfolio.max_leverage, 0.0)
        if requested_notional > max_notional:
            limit_scales.append(
                0.0 if requested_notional == 0.0 else max_notional / requested_notional
            )
    if portfolio.max_margin_usage is not None:
        max_margin = max(equity_cash * portfolio.max_margin_usage, 0.0)
        if requested_margin > max_margin:
            limit_scales.append(0.0 if requested_margin == 0.0 else max_margin / requested_margin)
    return limit_scales
