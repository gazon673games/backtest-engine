from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest
from test_multi_instrument_support import write_intraday_dataset, write_strategy

from backtest_engine.portfolio.multi.engine import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentExecutionConfig,
    MultiInstrumentLegConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
    run_multi_instrument_backtest,
)


def test_delayed_execution_does_not_expose_future_equity_to_strategy(tmp_path) -> None:
    dataset = write_intraday_dataset(
        tmp_path,
        "asset.parquet",
        "A",
        [100.0, 110.0, 120.0],
    )
    frame = pl.read_parquet(dataset)
    frame = pl.concat(
        [
            frame,
            frame.tail(1).with_columns(
                pl.lit(datetime(2024, 1, 2, 10, 15)).alias("ts"),
                pl.lit(130.0).alias("open"),
                pl.lit(130.0).alias("high"),
                pl.lit(130.0).alias("low"),
                pl.lit(130.0).alias("close"),
            ),
        ]
    )
    frame.write_parquet(dataset)
    plugin = write_strategy(
        tmp_path,
        """
def generate_targets(ctx):
    target = 1.0 if ctx.equity_cash <= 1000.0 else -1.0
    return {"A": {"target": target, "features": {"seen_equity": ctx.equity_cash}}}
""".lstrip(),
    )

    result = run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=[
                MultiInstrumentLegConfig(
                    asset_id="A",
                    dataset_path=dataset,
                    quality_approval_required=False,
                    allow_legacy_bar_timestamps=True,
                )
            ],
            strategy=MultiInstrumentStrategyConfig(strategy_file=plugin),
            portfolio=MultiInstrumentPortfolioConfig(initial_capital=1000.0),
            execution=MultiInstrumentExecutionConfig(
                timing="signal_close_to_next_bar_close",
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "out", run_id="temporal"),
        )
    )
    legs = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")

    assert legs.get_column("target").to_list() == [1.0, 1.0]
    assert result.metrics["final_equity_cash"] == pytest.approx(1020.0)
