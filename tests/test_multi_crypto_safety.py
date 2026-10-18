from __future__ import annotations

from types import SimpleNamespace

import pytest

from backtest_engine.accounting.cross_margin import CrossMarginAccount
from backtest_engine.accounting.perpetual import MarginTier, TieredMarginSchedule
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.portfolio.multi.accounting import validate_crypto_accounting_topology
from backtest_engine.portfolio.multi.config import (
    MultiInstrumentLegConfig,
    MultiInstrumentPortfolioConfig,
)


def perpetual_leg(tmp_path, asset_id: str) -> MultiInstrumentLegConfig:
    return MultiInstrumentLegConfig(
        asset_id=asset_id,
        dataset_path=tmp_path / f"{asset_id}.parquet",
        quality_approval_required=False,
        allow_legacy_bar_timestamps=True,
        asset_class="crypto",
        instrument_type="perpetual",
        pnl_currency="USDT",
        settlement_currency="USDT",
    )


def test_multi_margin_mode_validator_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Invalid margin mode"):
        MultiInstrumentPortfolioConfig(margin_mode="portfolio")


def test_multiple_perpetual_legs_enable_shared_cross_margin(tmp_path) -> None:
    loaded = {
        asset_id: {"config": perpetual_leg(tmp_path, asset_id)}
        for asset_id in ("BTCUSDT", "ETHUSDT")
    }

    config = SimpleNamespace(portfolio=MultiInstrumentPortfolioConfig(margin_mode="cross"))
    assert validate_crypto_accounting_topology(loaded, config) is True


def test_multiple_perpetual_legs_reject_isolated_mode(tmp_path) -> None:
    loaded = {
        asset_id: {"config": perpetual_leg(tmp_path, asset_id)}
        for asset_id in ("BTCUSDT", "ETHUSDT")
    }
    config = SimpleNamespace(portfolio=MultiInstrumentPortfolioConfig(margin_mode="isolated"))
    with pytest.raises(DatasetValidationError, match="margin_mode=cross"):
        validate_crypto_accounting_topology(loaded, config)


def test_cross_margin_offsets_pnl_and_fails_combined_initial_margin() -> None:
    schedule = TieredMarginSchedule([MarginTier(0.0, None, 0.1)])
    account = CrossMarginAccount(wallet_balance=100.0)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        account.register(
            symbol,
            point_value=1.0,
            leverage=10.0,
            margin_schedule=schedule,
            liquidation_fee_bps=1.0,
        )
    transitions = account.apply_batch(
        [
            {
                "symbol": "BTCUSDT",
                "quantity_delta": 1.0,
                "fill_price": 100.0,
                "fee_bps": 0.0,
                "funding_rate": 0.0,
            },
            {
                "symbol": "ETHUSDT",
                "quantity_delta": 5.0,
                "fill_price": 20.0,
                "fee_bps": 0.0,
                "funding_rate": 0.0,
            },
        ],
        {"BTCUSDT": 100.0, "ETHUSDT": 20.0},
    )
    assert len(transitions) == 2
    assert account.equity({"BTCUSDT": 50.0, "ETHUSDT": 30.0}) == 100.0
    assert account.liquidate_if_required({"BTCUSDT": 50.0, "ETHUSDT": 30.0}) == []

    constrained = CrossMarginAccount(wallet_balance=100.0)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        constrained.register(
            symbol,
            point_value=1.0,
            leverage=2.0,
            margin_schedule=schedule,
            liquidation_fee_bps=0.0,
        )
    with pytest.raises(DatasetValidationError, match="shared cross-margin collateral"):
        constrained.apply_batch(
            [
                {
                    "symbol": "BTCUSDT",
                    "quantity_delta": 1.0,
                    "fill_price": 100.0,
                    "fee_bps": 0.0,
                    "funding_rate": 0.0,
                },
                {
                    "symbol": "ETHUSDT",
                    "quantity_delta": 10.0,
                    "fill_price": 20.0,
                    "fee_bps": 0.0,
                    "funding_rate": 0.0,
                },
            ],
            {"BTCUSDT": 100.0, "ETHUSDT": 20.0},
        )
