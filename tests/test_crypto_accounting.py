from __future__ import annotations

from datetime import datetime

import pytest

from backtest_engine.accounting.crypto import settle_crypto_interval
from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime
from backtest_engine.accounting.perpetual import (
    LinearPerpetualAccount,
    MarginTier,
    TieredMarginSchedule,
)
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.intraday.event.config import IntradayEventInstrumentConfig
from backtest_engine.intraday.event.data import infer_dataset_instrument


def test_linear_perpetual_applies_fee_and_long_funding() -> None:
    settlement = settle_crypto_interval(
        instrument_type="perpetual",
        contract_type="linear",
        quantity=2.0,
        quantity_delta=2.0,
        point_value=1.0,
        fill_price=100.0,
        exit_price=110.0,
        low_price=99.0,
        high_price=112.0,
        equity=1_000.0,
        fee_bps=10.0,
        funding_rate=0.001,
        maintenance_margin_rate=0.005,
        liquidation_fee_bps=50.0,
    )

    assert settlement.trading_fee == pytest.approx(0.2)
    assert settlement.funding_payment == pytest.approx(0.22)
    assert settlement.maintenance_margin == pytest.approx(1.1)
    assert not settlement.liquidated


def test_short_receives_positive_funding_and_can_be_liquidated() -> None:
    settlement = settle_crypto_interval(
        instrument_type="perpetual",
        contract_type="linear",
        quantity=-1.0,
        quantity_delta=-1.0,
        point_value=1.0,
        fill_price=100.0,
        exit_price=105.0,
        low_price=95.0,
        high_price=150.0,
        equity=10.0,
        fee_bps=0.0,
        funding_rate=0.001,
        maintenance_margin_rate=0.1,
        liquidation_fee_bps=50.0,
    )

    assert settlement.funding_payment < 0.0
    assert settlement.liquidated
    assert settlement.liquidation_price == pytest.approx(100.0)
    assert settlement.liquidation_fee == pytest.approx(0.5)


def test_spot_rejects_short_position() -> None:
    with pytest.raises(DatasetValidationError, match="cannot hold a negative"):
        settle_crypto_interval(
            instrument_type="spot",
            contract_type="spot",
            quantity=-1.0,
            quantity_delta=-1.0,
            point_value=1.0,
            fill_price=100.0,
            exit_price=99.0,
            low_price=98.0,
            high_price=101.0,
            equity=100.0,
            fee_bps=10.0,
            funding_rate=0.0,
            maintenance_margin_rate=None,
            liquidation_fee_bps=0.0,
        )


def test_negative_net_cost_is_credited_by_ledger_runtime() -> None:
    runtime = BacktestLedgerRuntime(
        initial_capital=100.0,
        base_currency="USDT",
        start_ts=datetime(2024, 1, 1),
    )

    equity = runtime.apply_event_cash_flow(
        ts=datetime(2024, 1, 2),
        symbol="BTCUSDT",
        currency="USDT",
        gross_pnl=0.0,
        cost_cash=-1.0,
    )

    assert equity == pytest.approx(101.0)


def test_crypto_instrument_identity_is_inferred_from_dataset() -> None:
    instrument = infer_dataset_instrument(
        IntradayEventInstrumentConfig(),
        {
            "asset_class": "crypto",
            "instrument_type": "perpetual",
            "base_currency": "BTC",
            "quote_currency": "USDT",
            "settlement_currency": "USDT",
            "currency": "USDT",
        },
    )

    assert instrument.asset_class == "crypto"
    assert instrument.instrument_type == "perpetual"
    assert instrument.pnl_currency == "USDT"
    assert instrument.contract_type == "linear"


def test_spot_ledger_books_inventory_and_triangulated_equity() -> None:
    runtime = BacktestLedgerRuntime(
        initial_capital=1_000.0,
        base_currency="USD",
        fx_rates={"USDT/USD": 1.0},
        start_ts=datetime(2024, 1, 1),
        collateral_currency="USDT",
        initial_collateral=1_000.0,
    )

    equity = runtime.apply_spot_interval(
        fill_ts=datetime(2024, 1, 1, 0, 1),
        valuation_ts=datetime(2024, 1, 1, 0, 2),
        symbol="BTCUSDT",
        base_currency="BTC",
        quote_currency="USDT",
        quantity_delta=0.01,
        fill_price=50_000.0,
        mark_price=51_000.0,
        trading_fee_quote=0.5,
        additional_cost_quote=0.0,
        previous_equity_cash=1_000.0,
    )

    assert runtime.ledger.cash["BTC"] == pytest.approx(0.01)
    assert runtime.ledger.cash["USDT"] == pytest.approx(499.5)
    assert equity == pytest.approx(1_009.5)


def test_spot_ledger_rejects_unfunded_purchase_and_short_sale() -> None:
    runtime = BacktestLedgerRuntime(
        initial_capital=100.0,
        base_currency="USDT",
        start_ts=datetime(2024, 1, 1),
    )

    with pytest.raises(DatasetValidationError, match="unavailable inventory"):
        runtime.ledger.apply_spot_trade(
            ts=datetime(2024, 1, 1),
            symbol="BTCUSDT",
            side="buy",
            quantity=2.0,
            price=100.0,
            base_currency="BTC",
            quote_currency="USDT",
        )
    with pytest.raises(DatasetValidationError, match="unavailable inventory"):
        runtime.ledger.apply_spot_trade(
            ts=datetime(2024, 1, 1),
            symbol="BTCUSDT",
            side="sell",
            quantity=1.0,
            price=100.0,
            base_currency="BTC",
            quote_currency="USDT",
        )


def test_tiered_perpetual_account_keeps_entry_and_partially_liquidates() -> None:
    account = LinearPerpetualAccount(
        wallet_balance=100.0,
        point_value=1.0,
        leverage=10.0,
        margin_schedule=TieredMarginSchedule(
            [
                MarginTier(0.0, 1_000.0, 0.005),
                MarginTier(1_000.0, None, 0.01, 5.0),
            ]
        ),
        liquidation_fee_bps=50.0,
    )
    account.apply_fill(10.0, 100.0, 0.0)
    account.apply_fill(10.0, 110.0, 0.0)

    liquidation_price = account.liquidation_price()
    assert account.entry_price == pytest.approx(105.0)
    assert liquidation_price is not None
    result = account.liquidate_if_crossed(
        mark_low=liquidation_price - 1.0,
        mark_high=120.0,
        quantity_step=1.0,
        minimum_close_fraction=0.25,
    )

    assert result.liquidated
    assert result.status in {"partial", "full"}
    assert result.closed_quantity > 0.0
    assert result.bankruptcy_price is not None
