"""Regression checks for cross-leg causality and unsupported native payouts."""

import json
from datetime import datetime, timedelta

import polars as pl
import pytest
from neutral_adapter_fixture import NeutralAdapter, run_request

from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime
from backtest_engine.accounting.perpetual_inputs import perpetual_margin_schedule
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.instrument_registry import InstrumentSpec
from backtest_engine.workbench.datasets import register_workbench_dataset
from backtest_engine.workbench.runner import run_workbench_backtest


def test_two_legs_do_not_expose_first_legs_future_variation_at_second_fill():
    start = datetime(2024, 1, 1)
    fill, mark = start + timedelta(minutes=5), start + timedelta(minutes=10)
    ledger = BacktestLedgerRuntime(initial_capital=1000, base_currency="USD", start_ts=start)
    for symbol, pnl, fee in (("A", 100, 1), ("B", 200, 2)):
        ledger.apply_event_cash_flow(
            fill_ts=fill, ts=mark, symbol=symbol, currency="USD", gross_pnl=pnl, cost_cash=fee
        )
    assert ledger.decision_equity_at(start) == 1000
    assert ledger.decision_equity_at(fill) == 997
    assert ledger.decision_equity_at(mark) == 1297


def test_collateral_and_initial_nav_must_agree():
    with pytest.raises(DatasetValidationError, match="initial_collateral value"):
        BacktestLedgerRuntime(
            initial_capital=1000,
            base_currency="USD",
            fx_rates={"BTC/USD": 40000},
            collateral_currency="BTC",
            initial_collateral=0.1,
            start_ts=datetime(2024, 1, 1),
        )


@pytest.mark.parametrize("contract_type", ["inverse", "quanto"])
@pytest.mark.parametrize("product", ["cash", "dated", "swap"])
def test_unsupported_payout_cannot_silently_produce_a_linear_result(
    tmp_path, contract_type, product
):
    build = DatasetBuildRequest(
        adapter_id=NeutralAdapter.id,
        engine=product,
        security="GOLDUSD",
        from_date="2024-01-02",
        till_date="2024-01-02",
        output_root=tmp_path / "datasets",
    )
    dataset = NeutralAdapter().build_dataset(build)
    rules = json.loads(dataset.instrument_registry_path.read_text())
    rules["instruments"][0]["contract_type"] = contract_type
    dataset.instrument_registry_path.write_text(json.dumps(rules))
    dataset = register_workbench_dataset(tmp_path / "runs", build, dataset)
    with pytest.raises(DatasetValidationError, match=f"unsupported {contract_type}"):
        run_workbench_backtest(run_request(tmp_path, dataset))
    assert not list((tmp_path / "runs").rglob("events.parquet"))


@pytest.mark.parametrize("contract_type", ["inverse", "quanto"])
def test_stateful_perpetual_input_rejects_unsupported_models(contract_type):
    spec = InstrumentSpec(symbol="TEST", instrument_type="perpetual", contract_type=contract_type)
    with pytest.raises(DatasetValidationError, match=f"unsupported {contract_type}"):
        perpetual_margin_schedule(spec, require_margin_tiers=False)


def test_no_future_exchange_rate_is_used_for_decision_equity(tmp_path):
    from backtest_engine.accounting.historical_rates import HistoricalRateBook

    start = datetime(2024, 1, 1)
    later = start + timedelta(minutes=5)
    path = tmp_path / "fx.parquet"
    pl.DataFrame(
        {
            "ts": [start, later],
            "source_currency": ["USD", "USD"],
            "target_currency": ["EUR", "EUR"],
            "rate": [0.9, 0.5],
        }
    ).write_parquet(path)
    ledger = BacktestLedgerRuntime(
        initial_capital=900,
        base_currency="EUR",
        start_ts=start,
        initial_collateral=1000,
        collateral_currency="USD",
        fx_rates={"USD/EUR": 0.9},
        rate_book=HistoricalRateBook.from_path(path),
    )
    ledger.activate_rates(later)
    assert ledger.decision_equity_at(start) == 900
    assert ledger.decision_equity_at(later) == 500
