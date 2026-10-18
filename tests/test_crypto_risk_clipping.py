"""Risk-capped native orders must retain the instrument's executable quantity step."""

from datetime import timedelta

import polars as pl
import pytest
from crypto_return_fixture import SCRIPTED_ORDERS, START, SolReturnTransport, write_return_rules
from strategy_sdk.catalog_sources import snapshot_from_files

from backtest_engine.adapters.binance.adapter import BinanceAdapter
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.intraday.event._futures_sizing import apply_risk_controls
from backtest_engine.intraday.event.config import (
    IntradayEventInstrumentSpec,
    IntradayEventPortfolioConfig,
)
from backtest_engine.intraday.event.crypto_rules import validate_crypto_fill
from backtest_engine.workbench.models import WorkbenchRunRequest
from backtest_engine.workbench.runner import run_workbench_backtest


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize(
    ("step", "cap", "expected"),
    [
        (0.1, 18.98794703391013, 18.9),
        (1.0, 1.9659153649790824, 1.0),
        (0.001, 1.9659153649790824, 1.965),
        (1.0, 0.5, 0.0),
    ],
)
def test_clipped_target_and_reduction_obey_quantity_step(direction, step, cap, expected):
    instrument = IntradayEventInstrumentSpec(
        symbol="TESTUSDT",
        instrument_type="perpetual",
        quantity_step=step,
        minimum_quantity=step,
        pnl_currency="USDT",
    )
    portfolio = IntradayEventPortfolioConfig(
        initial_capital=1000,
        base_currency="USDT",
        max_position_notional=cap,
        risk_policy="clip",
        allow_fractional_contracts=True,
    )
    quantity = apply_risk_controls(direction * 20, 1, instrument, portfolio, 1000)
    assert quantity == pytest.approx(direction * expected)
    assert abs(quantity) <= cap
    validate_crypto_fill(instrument, direction * 20, quantity, 1, 1)


class GappedReturnTransport(SolReturnTransport):
    def get(self, product, path, params=None):
        rows = super().get(product, path, params)
        if path == "/fapi/v1/klines" or path == "/api/v3/klines":
            for row in rows:
                if row[0] == int((START + timedelta(hours=1)).timestamp() * 1000):
                    row[1] = "21"
        return rows


@pytest.mark.parametrize("product", ["spot", "perpetual"])
@pytest.mark.parametrize("gap", [False, True])
def test_native_risk_clipped_sdk_orders_execute_and_reconcile(tmp_path, product, gap):
    transport = GappedReturnTransport() if gap else SolReturnTransport()
    dataset = BinanceAdapter(transport).build_dataset(
        DatasetBuildRequest(
            adapter_id="binance",
            engine=product,
            market="crypto",
            security="SOLUSDT",
            from_date="2024-04-01",
            till_date="2024-04-01",
            timeframe="60m",
            rules_path=write_return_rules(tmp_path, product),
            rules_enforcement="strict",
            history_source="rest",
            gap_policy="reject",
            output_root=tmp_path / "datasets",
        )
    )
    snapshot = snapshot_from_files({"risk_step_fixture": SCRIPTED_ORDERS}, "risk_step_fixture")
    response = run_workbench_backtest(
        WorkbenchRunRequest(
            dataset_path=dataset.dataset_path,
            quality_approval_path=dataset.approval_path,
            script_source="SDK_CATALOG_SNAPSHOT = " + repr(snapshot) + "\n",
            script_language="python",
            strict_runtime=True,
            instrument_registry_path=dataset.instrument_registry_path,
            calendar_path=dataset.calendar_path,
            initial_capital=1000,
            base_currency="USDT",
            from_date="2024-04-01",
            max_position_notional=45.023,
            risk_policy="clip",
            execution_timing="signal_close_to_next_bar_open_to_close",
            end_of_data_position_policy="reject",
            output_root=tmp_path / "runs",
            run_id="risk-step",
        )
    )
    events = pl.read_parquet(response.output_dir / "events.parquet")
    ledger = pl.read_parquet(response.output_dir / "portfolio_ledger.parquet")
    fills = events.filter(pl.col("contract_delta") != 0)
    # The unchanged hold target is also capped after the closes at 24 and 12.
    expected_first = 2.143 if gap else 2.251
    assert fills["target_contracts"].to_list() == pytest.approx(
        [expected_first, 1.875, 1.125, 4.502, 3.751, 0]
    )
    assert ledger["reconciliation_diff"].abs().max() <= 1e-9
    assert ledger["equity_cash"][-1] - 1000 == pytest.approx(events["net_pnl"].sum())
