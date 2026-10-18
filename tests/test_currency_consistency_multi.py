from datetime import date

import polars as pl
import pytest
from test_multi_instrument_support import (
    write_intraday_dataset as write_multi_dataset,
)
from test_multi_instrument_support import write_strategy

from backtest_engine.portfolio.multi.engine import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentExecutionConfig,
    MultiInstrumentLegConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
    run_multi_instrument_backtest,
)


def test_multi_currency_equity_metrics_and_risk_use_base_currency(tmp_path) -> None:
    rub_path = write_multi_dataset(tmp_path, "rub.parquet", "RUB_LEG", [100.0, 101.0])
    usd_path = write_multi_dataset(tmp_path, "usd.parquet", "USD_LEG", [100.0, 101.0])
    plugin = write_strategy(
        tmp_path,
        'def generate_targets(ctx):\n    return {"RUB_LEG": 1, "USD_LEG": 1}\n',
    )
    result = run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=[
                MultiInstrumentLegConfig(
                    asset_id="RUB_LEG",
                    dataset_path=rub_path,
                    quality_approval_required=False,
                    allow_legacy_bar_timestamps=True,
                ),
                MultiInstrumentLegConfig(
                    asset_id="USD_LEG",
                    dataset_path=usd_path,
                    price_currency="USD",
                    pnl_currency="USD",
                    initial_margin=10.0,
                    quality_approval_required=False,
                    allow_legacy_bar_timestamps=True,
                ),
            ],
            strategy=MultiInstrumentStrategyConfig(strategy_file=plugin),
            portfolio=MultiInstrumentPortfolioConfig(
                initial_capital=10_000.0,
                base_currency="RUB",
                fx_rates={"USD/RUB": 90.0},
                sizing_mode="notional",
                target_notional=18_000.0,
                max_position_notional=9_000.0,
                risk_policy="clip",
            ),
            execution=MultiInstrumentExecutionConfig(
                timing="signal_close_to_same_close",
                allow_noncausal_same_close=True,
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "out", run_id="multi_fx"),
        )
    )
    legs = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    equity = pl.read_parquet(result.output_dir / "portfolio_equity.parquet")
    ledger = pl.read_parquet(result.output_dir / "portfolio_ledger.parquet")
    first_equity = equity.row(0, named=True)

    assert legs.filter(pl.col("asset_id") == "RUB_LEG")["target_contracts"][0] == 90.0
    usd_leg = legs.filter(pl.col("asset_id") == "USD_LEG").row(0, named=True)
    assert usd_leg["target_contracts"] == 1.0
    assert usd_leg["net_return"] == pytest.approx(0.009)
    assert usd_leg["margin_usage"] == pytest.approx(0.09)
    assert first_equity["gross_position_notional"] == pytest.approx(18_000.0)
    assert first_equity["gross_pnl"] == pytest.approx(180.0)
    assert first_equity["net_pnl"] == pytest.approx(180.0)
    assert first_equity["net_return"] == pytest.approx(0.018)
    assert result.metrics["final_equity_cash"] == pytest.approx(10_180.0)
    assert result.metrics["net_pnl"] == pytest.approx(180.0)
    assert ledger.get_column("equity_cash")[-1] == pytest.approx(10_180.0)
