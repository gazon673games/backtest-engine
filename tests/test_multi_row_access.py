from datetime import date, datetime, timedelta

import polars as pl
from polars.testing import assert_frame_equal
from test_multi_instrument_support import write_intraday_dataset

from backtest_engine.portfolio.multi.config import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentExecutionConfig,
    MultiInstrumentInstrumentSpec,
    MultiInstrumentLegConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
)
from backtest_engine.portfolio.multi.data import load_legs
from backtest_engine.portfolio.multi.events import build_multi_instrument_events


def test_lazy_leg_rows_preserve_dated_specs_and_every_ledger_entry(tmp_path):
    path = write_intraday_dataset(tmp_path, "asset.parquet", "A", [100.0])
    count = 100
    source = pl.read_parquet(path)
    frame = pl.concat([source] * count).with_columns(
        pl.Series(
            "ts",
            [
                datetime(2024, 1, 2, 10) + timedelta(days=i // 50, minutes=5 * (i % 50))
                for i in range(count)
            ],
        ),
        *[
            pl.Series(column, [100.0 + i % 10 for i in range(count)])
            for column in ("open", "high", "low", "close")
        ],
    )
    frame.write_parquet(path)
    leg = MultiInstrumentLegConfig(
        asset_id="A",
        dataset_path=path,
        quality_approval_required=False,
        allow_legacy_bar_timestamps=True,
        specs=[
            MultiInstrumentInstrumentSpec(symbol="A", valid_to=date(2024, 1, 2)),
            MultiInstrumentInstrumentSpec(
                symbol="A",
                valid_from=date(2024, 1, 3),
                contract_multiplier=2.0,
            ),
        ],
    )
    config = MultiInstrumentBacktestConfig(
        legs=[leg],
        strategy=MultiInstrumentStrategyConfig(strategy_file=tmp_path / "strategy.py"),
        portfolio=MultiInstrumentPortfolioConfig(base_currency="RUB", initial_capital=10000.0),
        execution=MultiInstrumentExecutionConfig(
            timing="signal_close_to_same_close",
            allow_noncausal_same_close=True,
            from_date=date(2024, 1, 2),
            end_of_data_position_policy="mark",
            commission_bps=1.0,
        ),
        output=MultiInstrumentOutputConfig(root=tmp_path / "runs", run_id="row_access"),
    )
    loaded = load_legs([leg], config.execution.from_date)
    lazy_rows = loaded["A"]["rows"]

    def targets(ctx):
        return {"A": 1 if ctx.current_ts.date() == date(2024, 1, 2) else -1}

    actual = build_multi_instrument_events(config, loaded, targets)
    loaded["A"]["rows"] = frame.to_dicts()
    expected = build_multi_instrument_events(config, loaded, targets)
    for actual_frame, expected_frame in zip(actual[:3], expected[:3], strict=True):
        assert_frame_equal(actual_frame, expected_frame, check_exact=True)
    assert actual[3] == expected[3]
    assert lazy_rows.cached_row_count <= 32
