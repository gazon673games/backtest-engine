"""Lot-safe liquidations, fee-aware additions and accurate SDK fill transitions."""

from datetime import datetime, timedelta

import pytest
from test_intraday_event_support import make_config

from backtest_engine.accounting.perpetual import (
    LinearPerpetualAccount,
    MarginTier,
    TieredMarginSchedule,
)
from backtest_engine.intraday.event.add_affordability import skip_unaffordable_add
from backtest_engine.intraday.event.config import IntradayEventInstrumentSpec
from backtest_engine.intraday.event.lifecycle import notify_event_fill
from backtest_engine.strategies.plugins.normalization import IntradayEventPluginDecision


@pytest.mark.parametrize("quantity,step", [(1111.0, 1.0), (-1111.0, 1.0), (1.111, 0.001)])
def test_partial_liquidation_and_residual_obey_quantity_step(quantity, step):
    account = LinearPerpetualAccount(
        wallet_balance=abs(quantity) * 100 / 3,
        point_value=1,
        leverage=3,
        margin_schedule=TieredMarginSchedule([MarginTier(0, None, 0.005)]),
        liquidation_fee_bps=0,
    )
    account.apply_fill(quantity, 100, 0)
    price = account.liquidation_price()
    result = account.liquidate_if_crossed(
        mark_low=price - 1, mark_high=price + 1, quantity_step=step
    )
    assert result.liquidated and result.status == "partial"
    assert result.closed_quantity / step == pytest.approx(round(result.closed_quantity / step))
    assert result.remaining_quantity / step == pytest.approx(
        round(result.remaining_quantity / step)
    )
    assert result.closed_quantity >= abs(quantity) * 0.25
    account.apply_fill(-result.remaining_quantity, price, 5)
    assert account.quantity == pytest.approx(0)


@pytest.mark.parametrize("capital,expected", [(500.0, True), (501.0, False)])
@pytest.mark.parametrize("side", [-1, 1])
def test_perpetual_add_reserves_native_margin_and_taker_fee(tmp_path, capital, expected, side):
    config = make_config(
        tmp_path,
        tmp_path / "bars.parquet",
        tmp_path / "strategy.py",
        sizing_mode="strategy_contracts",
        initial_capital=capital,
    )
    config.portfolio.leverage = 2
    config.portfolio.base_currency = "USDT"
    instrument = IntradayEventInstrumentSpec(
        symbol="TESTUSDT",
        instrument_type="perpetual",
        price_currency="USDT",
        pnl_currency="USDT",
        quantity_step=1,
        taker_fee_bps=10,
        initial_margin=None,
    )
    decision = IntradayEventPluginDecision(
        target=side, contracts=10, skip_unaffordable_add=True, reason="add"
    )
    state = {}
    skipped = skip_unaffordable_add(
        decision,
        previous_contracts=8 * side,
        equity_cash=capital,
        fill_price=100,
        fill_row=dict(ts=datetime(2026, 1, 1), open=100, high=100, low=100, close=100, volume=1000),
        instrument=instrument,
        config=config,
        decision_state=state,
    )
    assert skipped is expected


class FilledStrategy:
    def __init__(self):
        self.fills = []

    def notify_fill(self, event):
        self.fills.append(event)


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("remaining", [0, 75])
def test_order_then_liquidation_callbacks_preserve_positions_and_times(side, remaining):
    strategy = FilledStrategy()
    at = datetime(2026, 1, 1)
    event = dict(
        liquidated=True,
        previous_contracts=0,
        filled_contract_delta=100 * side,
        contract_delta=remaining * side,
        target_contracts=remaining * side,
        remaining_quantity=remaining * side,
        liquidation_quantity=100 - remaining,
        liquidation_price=80,
        fill_price=100,
        fill_ts=at,
        next_ts=at + timedelta(hours=12),
        order_id="entry",
    )
    notify_event_fill(strategy, event)
    assert len(strategy.fills) == 2
    entry, reduction = strategy.fills
    assert entry["target_contracts"] == 100 * side and not entry["liquidated"]
    assert entry["fill_ts"] == at
    assert reduction["target_contracts"] == remaining * side
    assert reduction["contract_delta"] == -(100 - remaining) * side
    assert reduction["liquidated"] is (remaining == 0)
    assert reduction["fill_ts"] == event["next_ts"]
    assert reduction["fill_price"] == 80
    assert reduction["fill_id"] != entry["fill_id"]
