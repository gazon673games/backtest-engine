"""Buy 2.5 at 100, sell at 120: profit 50 less the declared fees."""

import json
from datetime import datetime, timedelta

import polars as pl
import pytest
from neutral_adapter_fixture import NeutralAdapter, run_request

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.data.instrument_registry import InstrumentSpec
from backtest_engine.portfolio.multi.config import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentExecutionConfig,
    MultiInstrumentLegConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
)
from backtest_engine.portfolio.multi.engine import run_multi_instrument_backtest
from backtest_engine.workbench.datasets import register_workbench_dataset
from backtest_engine.workbench.runner import run_workbench_backtest


@pytest.mark.parametrize("product", ["cash", "dated", "swap"])
@pytest.mark.parametrize("historical_fx", [False, True])
@pytest.mark.parametrize(
    ("maker", "taker", "role", "expected"),
    [(0, 0, "taker", 1049.45), (0, 20, "maker", 1050), (0, 20, "taker", 1048.9)],
)
def test_same_fee_policy_in_single_and_portfolio(
    tmp_path, product, historical_fx, maker, taker, role, expected
):
    adapter = NeutralAdapter()
    build = DatasetBuildRequest(
        adapter_id=adapter.id,
        engine=product,
        security="GOLDUSD",
        from_date="2024-01-02",
        till_date="2024-01-02",
        output_root=tmp_path / "datasets",
    )
    dataset = adapter.build_dataset(build)
    rules = json.loads(dataset.instrument_registry_path.read_text())
    rules["instruments"][0].update(maker_fee_bps=maker, taker_fee_bps=taker)
    dataset.instrument_registry_path.write_text(json.dumps(rules))
    dataset = register_workbench_dataset(tmp_path / "runs", build, dataset)
    portfolio = {"initial_capital": 1000, "base_currency": "USD"}
    if product == "dated":
        expected = 1048.95  # Generic bps plus 0.1 USD per contract on each fill.
    if historical_fx:
        rates = tmp_path / "fx.parquet"
        pl.DataFrame(
            {
                "ts": [datetime(2024, 1, 2, 1, 5) + timedelta(minutes=5 * i) for i in range(6)],
                "source_currency": ["USD"] * 6,
                "target_currency": ["EUR"] * 6,
                "rate": [0.9, 0.8, 0.75, 0.7, 0.65, 0.6],
            }
        ).write_parquet(rates)
        portfolio.update(
            initial_capital=900,
            base_currency="EUR",
            collateral_currency="USD",
            initial_collateral=1000,
            valuation_rates_path=rates,
        )
        expected *= 0.6
    request = run_request(tmp_path, dataset).model_copy(
        update={"commission_bps": 10, "liquidity_role": role, **portfolio}
    )
    single = run_workbench_backtest(request)
    spec = InstrumentSpec.model_validate(rules["instruments"][0])
    strategy = tmp_path / "orders.py"
    strategy.write_text(
        "calls = 0\ndef generate_targets(ctx):\n    global calls\n    calls += 1\n"
        '    return {"GOLDUSD": 1 if calls < 3 else 0}\n'
    )
    multi = run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=[
                MultiInstrumentLegConfig(
                    asset_id=spec.symbol,
                    dataset_path=dataset.dataset_path,
                    quality_approval_required=False,
                    **spec.model_dump(exclude={"symbol"}),
                )
            ],
            strategy=MultiInstrumentStrategyConfig(strategy_file=strategy),
            portfolio=MultiInstrumentPortfolioConfig(
                **portfolio,
                fixed_contracts=2.5,
            ),
            execution=MultiInstrumentExecutionConfig(
                timing=request.execution_timing,
                from_date=request.from_date,
                commission_bps=10,
                liquidity_role=role,
                end_of_data_position_policy="mark",
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "runs", run_id="multi"),
        )
    )
    for result in (single, multi):
        ledger = pl.read_parquet(result.output_dir / "portfolio_ledger.parquet")
        assert ledger["equity_cash"][-1] == pytest.approx(expected, abs=1e-9)
        assert ledger["reconciliation_diff"].abs().max() == 0
