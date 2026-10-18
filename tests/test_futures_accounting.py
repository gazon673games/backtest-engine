from __future__ import annotations

import pytest

from backtest_engine.accounting.futures import (
    FuturesExecutionCostSpec,
    FuturesInstrumentSpec,
    FuturesPortfolioSpec,
    apply_contract_risk,
    contract_point_value,
    cost_breakdown,
    effective_tick_value,
    pnl_cash,
    position_notional,
    target_contracts,
    turnover_notional,
)


def test_futures_accounting_contract_values_and_pnl() -> None:
    instrument = FuturesInstrumentSpec(
        contract_multiplier=10.0,
        lot_size=2.0,
        tick_size=0.5,
    )

    assert contract_point_value(instrument) == 20.0
    assert effective_tick_value(instrument) == 10.0
    assert position_notional(3.0, 100.0, instrument) == 6000.0
    assert turnover_notional(-2.0, 100.0, instrument) == 4000.0
    assert (
        pnl_cash(contracts=3.0, fill_price=100.0, exit_price=101.0, instrument=instrument) == 60.0
    )


@pytest.mark.parametrize(
    ("instrument", "expected_point_value"),
    [
        (
            FuturesInstrumentSpec(
                symbol="SiU6",
                contract_multiplier=1000.0,
                tick_size=1.0,
                tick_value=1.0,
            ),
            1.0,
        ),
        (
            FuturesInstrumentSpec(
                symbol="CRU6",
                contract_multiplier=1000.0,
                tick_size=0.001,
                tick_value=1.0,
            ),
            1000.0,
        ),
    ],
)
def test_futures_accounting_prefers_authoritative_tick_ratio(
    instrument: FuturesInstrumentSpec,
    expected_point_value: float,
) -> None:
    assert contract_point_value(instrument) == expected_point_value
    assert effective_tick_value(instrument) == 1.0
    assert (
        pnl_cash(
            contracts=30.0,
            fill_price=100.0,
            exit_price=115.0,
            instrument=instrument,
        )
        == 30.0 * 15.0 * expected_point_value
    )


def test_futures_accounting_notional_sizing_and_risk_clip() -> None:
    instrument = FuturesInstrumentSpec(contract_multiplier=10.0, initial_margin=1000.0)
    portfolio = FuturesPortfolioSpec(
        initial_capital=10000.0,
        sizing_mode="notional",
        target_notional=25_000.0,
        allow_fractional_contracts=False,
        max_margin_usage=0.2,
        risk_policy="clip",
    )

    requested = target_contracts(
        signal=1.0,
        fill_price=100.0,
        instrument=instrument,
        portfolio=portfolio,
        equity_cash=10_000.0,
    )
    risk = apply_contract_risk(
        contracts=requested,
        fill_price=100.0,
        instrument=instrument,
        portfolio=portfolio,
        equity_cash=10_000.0,
    )

    assert requested == 25.0
    assert risk.status == "clipped"
    assert risk.contracts == 2.0


def test_futures_accounting_cost_breakdown_components() -> None:
    costs = cost_breakdown(
        contract_delta=2.0,
        turnover_notional_cash=10_000.0,
        current_contracts=1.0,
        roll_notional_cash=5_000.0,
        instrument=FuturesInstrumentSpec(tick_value=10.0),
        execution=FuturesExecutionCostSpec(
            commission_bps=1.0,
            slippage_bps=2.0,
            slippage_ticks=0.5,
            bid_ask_spread_bps=4.0,
            bid_ask_spread_ticks=1.0,
            roll_cost_bps=3.0,
            roll_slippage_ticks=0.25,
        ),
    )

    assert costs.commission_cost_cash == pytest.approx(1.0)
    assert costs.slippage_cost_cash == pytest.approx(12.0)
    assert costs.spread_cost_cash == pytest.approx(12.0)
    assert costs.roll_cost_cash == pytest.approx(4.0)
    assert costs.total_cost_cash == pytest.approx(29.0)


def test_futures_risk_accepts_contract_count_declared_by_strategy() -> None:
    result = apply_contract_risk(
        contracts=-3.0,
        fill_price=100.0,
        instrument=FuturesInstrumentSpec(contract_multiplier=10.0),
        portfolio=FuturesPortfolioSpec(
            initial_capital=10_000.0,
            sizing_mode="strategy_contracts",
            max_leverage=1.0,
            risk_policy="clip",
        ),
        equity_cash=10_000.0,
    )

    assert result.status == "ok"
    assert result.contracts == -3.0
