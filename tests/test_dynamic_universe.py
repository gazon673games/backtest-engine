from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from backtest_engine.data.universe import PointInTimeUniverse
from backtest_engine.portfolio.multi.engine import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentExecutionConfig,
    MultiInstrumentLegConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
    MultiInstrumentUniverseConfig,
    run_multi_instrument_backtest,
)


def test_point_in_time_universe_hides_future_publications() -> None:
    first = datetime(2025, 1, 1, tzinfo=UTC)
    second = first + timedelta(days=1)
    universe = PointInTimeUniverse(
        pl.DataFrame(
            {
                "effective_at": [first, first, second, second],
                "available_at": [first, first, second, second],
                "asset_id": ["A", "B", "A", "B"],
                "status": ["active"] * 4,
                "quote_volume": [100.0, 50.0, 10.0, 200.0],
            }
        )
    )

    before_rebalance = universe.select(first, top_n=1)
    after_rebalance = universe.select(second, top_n=1)

    assert before_rebalance.asset_ids == ("A",)
    assert after_rebalance.asset_ids == ("B",)


def test_multi_backtest_liquidates_asset_removed_from_universe(tmp_path: Path) -> None:
    start = datetime(2024, 1, 2, 10, 0)
    timestamps = [start + timedelta(minutes=5 * index) for index in range(4)]
    legs = [
        MultiInstrumentLegConfig(
            asset_id=asset_id,
            dataset_path=_write_leg(tmp_path, asset_id, timestamps),
            quality_approval_required=False,
            allow_legacy_bar_timestamps=True,
        )
        for asset_id in ("A", "B")
    ]
    universe_path = tmp_path / "universe.parquet"
    pl.DataFrame(
        {
            "effective_at": [start, start, timestamps[1], timestamps[1]],
            "available_at": [start, start, timestamps[1], timestamps[1]],
            "asset_id": ["A", "B", "A", "B"],
            "status": ["active"] * 4,
            "quote_volume": [100.0, 50.0, 10.0, 200.0],
        }
    ).write_parquet(universe_path)
    strategy = tmp_path / "strategy.py"
    strategy.write_text(
        "def generate_targets(ctx):\n"
        "    return {asset_id: 1.0 for asset_id in ctx.tradable_assets}\n",
        encoding="utf-8",
    )
    config = MultiInstrumentBacktestConfig(
        legs=legs,
        strategy=MultiInstrumentStrategyConfig(strategy_file=strategy),
        portfolio=MultiInstrumentPortfolioConfig(initial_capital=10_000.0),
        execution=MultiInstrumentExecutionConfig(
            timing="signal_close_to_next_bar_close",
            end_of_data_position_policy="mark",
        ),
        universe=MultiInstrumentUniverseConfig(path=universe_path, top_n=1),
        output=MultiInstrumentOutputConfig(root=tmp_path / "runs", run_id="universe"),
    )

    result = run_multi_instrument_backtest(config)
    rows = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    removal = rows.filter(
        (pl.col("asset_id") == "A") & (pl.col("strategy_reason") == "universe_removed")
    )

    assert removal.height == 1
    assert removal.get_column("target_contracts").to_list() == [0.0]
    assert rows.filter((pl.col("asset_id") == "B") & (pl.col("target_contracts") > 0.0)).height > 0


def _write_leg(
    root: Path,
    asset_id: str,
    timestamps: list[datetime],
) -> Path:
    path = root / f"{asset_id}.parquet"
    closes = [100.0, 101.0, 102.0, 103.0]
    pl.DataFrame(
        {
            "source": ["fixture"] * 4,
            "exchange": ["TEST"] * 4,
            "engine": ["futures"] * 4,
            "market": ["test"] * 4,
            "board": ["TEST"] * 4,
            "symbol": [asset_id] * 4,
            "continuous_symbol": [f"{asset_id}_CONT"] * 4,
            "timeframe": ["5m"] * 4,
            "ts": timestamps,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000.0] * 4,
            "value": [100_000.0] * 4,
            "currency": ["RUB"] * 4,
            "active_contract": [asset_id] * 4,
            "source_contract_symbol": [asset_id] * 4,
            "dataset_type": ["nearest_contract"] * 4,
            "adjustment_mode": ["unadjusted"] * 4,
        }
    ).write_parquet(path)
    return path
