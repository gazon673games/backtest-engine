"""The same recorded third-venue orders pass through the portfolio engine."""

from pathlib import Path

import polars as pl
import pytest
from neutral_adapter_fixture import NeutralAdapter

from backtest_engine.adapters.models import DatasetBuildRequest
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


@pytest.mark.parametrize(
    ("product", "direction", "expected"),
    [
        ("cash", 1, 1049.45),
        ("dated", 1, 1049.5),
        ("swap", 1, 1049.45),
        ("dated", -1, 949.5),
        ("swap", -1, 949.45),
    ],
)
def test_third_venue_portfolio_uses_the_same_currency_fees_and_marks(
    tmp_path: Path, product, direction, expected
):
    request = DatasetBuildRequest(
        adapter_id=NeutralAdapter.id,
        engine=product,
        security="GOLDUSD",
        from_date="2024-01-02",
        till_date="2024-01-02",
        output_root=tmp_path / "datasets",
    )
    dataset = register_workbench_dataset(
        tmp_path / "runs", request, NeutralAdapter().build_dataset(request)
    )
    script = tmp_path / "orders.py"
    script.write_text(
        """STRATEGY_METADATA = {"id": "neutral_multi_fixture"}
def initialize(params):
    return {"calls": 0}
def generate_targets(ctx):
    ctx.state["calls"] += 1
    return {"GOLDUSD": DIRECTION if ctx.state["calls"] < 3 else 0}
""".replace("DIRECTION", str(direction)),
        encoding="utf-8",
    )
    result = run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=[
                MultiInstrumentLegConfig(
                    asset_id="GOLDUSD",
                    catalog_path=dataset.catalog_path,
                    dataset_id=dataset.dataset_id,
                    catalog_enforcement="strict",
                )
            ],
            strategy=MultiInstrumentStrategyConfig(strategy_file=script, strict_runtime=True),
            portfolio=MultiInstrumentPortfolioConfig(
                initial_capital=1000,
                fixed_contracts=2.5,
                require_mark_price=True,
                require_margin_tiers=True,
            ),
            execution=MultiInstrumentExecutionConfig(
                timing="signal_close_to_next_bar_open_to_close"
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "runs", run_id="multi"),
        )
    )
    ledger = pl.read_parquet(result.output_dir / "portfolio_ledger.parquet")
    assert ledger["equity_cash"][-1] == pytest.approx(expected, abs=1e-9)
    assert set(ledger["currency"]) == {"USD"}
    assert result.metrics["total_return"] == pytest.approx(expected / 1000 - 1)
    assert result.metrics["bars_per_year"] == 260 * 5
    assert not (result.output_dir / "forts_accounting.parquet").exists()
