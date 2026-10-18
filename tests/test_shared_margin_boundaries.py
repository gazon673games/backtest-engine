"""Shared perpetual collateral must reconcile native PnL and time-specific FX."""

import json
from datetime import date, datetime, timedelta

import polars as pl
import pytest
from neutral_adapter_fixture import NeutralAdapter

from backtest_engine.accounting.cross_margin import CrossMarginAccount
from backtest_engine.accounting.perpetual import MarginTier, TieredMarginSchedule
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.portfolio.multi.config import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentExecutionConfig,
    MultiInstrumentLegConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
)
from backtest_engine.portfolio.multi.engine import run_multi_instrument_backtest


@pytest.mark.parametrize("historical_fx", [False, True])
@pytest.mark.parametrize("product", ["cash", "dated", "swap"])
def test_shared_wallet_buy_sell_and_fx_reconcile(tmp_path, historical_fx, product):
    dataset = NeutralAdapter().build_dataset(
        DatasetBuildRequest(
            adapter_id=NeutralAdapter.id,
            engine=product,
            security="GOLDUSD",
            from_date="2024-01-02",
            till_date="2024-01-02",
            output_root=tmp_path / "data",
        )
    )
    spec = json.loads(dataset.instrument_registry_path.read_text())["instruments"][0]
    spec.pop("symbol")
    spec["taker_fee_bps"] = 0
    legs = []
    for symbol in ("A", "B"):
        path = tmp_path / f"{symbol}.parquet"
        pl.read_parquet(dataset.dataset_path).with_columns(
            pl.lit(symbol).alias("symbol"),
            pl.lit(symbol).alias("active_contract"),
            pl.lit(symbol).alias("base_currency"),
        ).write_parquet(path)
        legs.append(
            MultiInstrumentLegConfig(
                asset_id=symbol,
                dataset_path=path,
                quality_approval_required=False,
                **{**spec, "base_currency": symbol},
            )
        )
    portfolio = {"initial_capital": 1000, "base_currency": "USD"}
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
            initial_collateral=1000,
            collateral_currency="USD",
            valuation_rates_path=rates,
        )
    strategy = tmp_path / "orders.py"
    strategy.write_text(
        "calls=0\ndef generate_targets(ctx):\n    global calls\n    calls+=1\n"
        "    return {key: 1 if calls<3 else 0 for key in ctx.tradable_assets}\n"
    )
    result = run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=legs,
            strategy=MultiInstrumentStrategyConfig(strategy_file=strategy),
            portfolio=MultiInstrumentPortfolioConfig(**portfolio, fixed_contracts=2.5),
            execution=MultiInstrumentExecutionConfig(
                timing="signal_close_to_next_bar_open_to_close",
                commission_bps=10,
                from_date=date(2024, 1, 2),
                end_of_data_position_policy="mark",
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "runs", run_id="shared"),
        )
    )
    ledger = pl.read_parquet(result.output_dir / "portfolio_ledger.parquet")
    rows = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    expected = (1097.9 if product == "dated" else 1098.9) * (0.6 if historical_fx else 1)
    assert ledger["equity_cash"][-1] == pytest.approx(expected)
    if product != "dated":
        assert rows["trading_fee_cash"].sum() == pytest.approx(1.1)
    assert rows["net_pnl_base"].sum() == pytest.approx(expected - portfolio["initial_capital"])
    equity = pl.read_parquet(result.output_dir / "portfolio_equity.parquet")
    first = rows.filter(pl.col("ts") == rows["ts"][0])
    assert first["net_return"].sum() == pytest.approx(equity["net_return"][0])


def test_future_funding_cannot_finance_initial_margin():
    account = CrossMarginAccount(5)
    account.register(
        "A",
        point_value=1,
        leverage=10,
        margin_schedule=TieredMarginSchedule([MarginTier(0, None, 0.005)]),
        liquidation_fee_bps=0,
    )
    with pytest.raises(DatasetValidationError, match="shared cross-margin collateral"):
        account.apply_batch(
            [
                {
                    "symbol": "A",
                    "quantity_delta": 1,
                    "fill_price": 100,
                    "fee_bps": 0,
                    "funding_payment": -100,
                }
            ],
            {"A": 100},
        )
    assert account.wallet_balance == 5
    assert account.positions["A"].quantity == 0
