# ruff: noqa
import json
import math

from test_multi_instrument_support import *  # noqa: F401,F403


def test_multi_instrument_portfolio_runs_two_assets_on_shared_cash(tmp_path: Path) -> None:
    asset_a = write_intraday_dataset(tmp_path, "asset_a.parquet", "A", [100.0, 110.0, 105.0])
    asset_b = write_intraday_dataset(tmp_path, "asset_b.parquet", "B", [50.0, 45.0, 40.0])
    plugin = write_strategy(
        tmp_path,
        """
def generate_targets(ctx, params=None):
    return {
        "A": {"target": 1.0, "reason": "long_a"},
        "B": {"target": -1.0, "reason": "short_b"},
    }
""".lstrip(),
    )

    result = run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=[
                MultiInstrumentLegConfig(
                    asset_id="A",
                    dataset_path=asset_a,
                    quality_approval_required=False,
                    allow_legacy_bar_timestamps=True,
                ),
                MultiInstrumentLegConfig(
                    asset_id="B",
                    dataset_path=asset_b,
                    quality_approval_required=False,
                    allow_legacy_bar_timestamps=True,
                ),
            ],
            strategy=MultiInstrumentStrategyConfig(strategy_file=plugin),
            portfolio=MultiInstrumentPortfolioConfig(initial_capital=1000.0),
            execution=MultiInstrumentExecutionConfig(
                trading_days_per_year=252,
                timing="signal_close_to_same_close",
                allow_noncausal_same_close=True,
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "out", run_id="multi"),
        )
    )

    legs = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    equity = pl.read_parquet(result.output_dir / "portfolio_equity.parquet")

    assert result.metrics["asset_count"] == 2
    assert result.metrics["leg_event_count"] == 4
    assert result.metrics["final_equity_cash"] == pytest.approx(1015.0)
    assert result.metrics["total_return"] == pytest.approx(0.015)
    assert legs.filter(pl.col("asset_id") == "A").get_column("gross_pnl").to_list() == [
        10.0,
        -5.0,
    ]
    assert legs.filter(pl.col("asset_id") == "B").get_column("gross_pnl").to_list() == [
        5.0,
        5.0,
    ]
    assert equity.get_column("net_pnl").to_list() == [15.0, 0.0]
    assert result.metrics["bars_per_trading_day"] == 2.0
    assert result.metrics["sharpe"] == pytest.approx(math.sqrt(252.0))


def test_mixed_portfolio_uses_real_spot_inventory_and_futures_cash(tmp_path: Path) -> None:
    spot = write_spot_dataset(tmp_path, "spot.parquet", "BTCUSDT", [100.0, 110.0, 120.0])
    future = write_intraday_dataset(tmp_path, "future.parquet", "FUT", [10.0, 11.0, 12.0])
    plugin = write_strategy(
        tmp_path,
        'def generate_targets(ctx):\n    return {"BTC": 1, "FUT": 1}\n',
    )

    result = run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=[
                MultiInstrumentLegConfig(
                    asset_id="BTC",
                    dataset_path=spot,
                    quality_approval_required=False,
                    allow_legacy_bar_timestamps=True,
                ),
                MultiInstrumentLegConfig(
                    asset_id="FUT",
                    dataset_path=future,
                    price_currency="USDT",
                    pnl_currency="USDT",
                    quality_approval_required=False,
                    allow_legacy_bar_timestamps=True,
                ),
            ],
            strategy=MultiInstrumentStrategyConfig(strategy_file=plugin),
            portfolio=MultiInstrumentPortfolioConfig(
                initial_capital=1000.0,
                base_currency="USDT",
            ),
            execution=MultiInstrumentExecutionConfig(
                trading_days_per_year=252,
                timing="signal_close_to_same_close",
                allow_noncausal_same_close=True,
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "out", run_id="mixed"),
        )
    )

    legs = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    ledger = pl.read_parquet(result.output_dir / "portfolio_ledger.parquet")
    assert result.metrics["final_equity_cash"] == pytest.approx(1022.0)
    assert legs.filter(pl.col("asset_id") == "BTC").get_column(
        "instrument_type"
    ).unique().to_list() == ["spot"]
    assert ledger.get_column("reconciliation_diff").fill_null(0.0).abs().max() == 0.0


def test_multi_strategy_lifecycle_receives_state_fills_and_end(tmp_path: Path) -> None:
    asset = write_intraday_dataset(tmp_path, "asset.parquet", "A", [100.0, 101.0, 102.0])
    lifecycle_output = tmp_path / "lifecycle.json"
    plugin = write_strategy(
        tmp_path,
        """
import json
from pathlib import Path

def initialize(params):
    return {"path": params["path"], "calls": 0, "fills": 0}

def generate_targets(ctx, params):
    ctx.state["calls"] += 1
    return {"A": 1.0}

def on_fill(state, event):
    state["fills"] += 1

def on_end(state, summary):
    Path(state["path"]).write_text(json.dumps({**state, **summary}), encoding="utf-8")
""".lstrip(),
    )

    run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=[
                MultiInstrumentLegConfig(
                    asset_id="A",
                    dataset_path=asset,
                    quality_approval_required=False,
                    allow_legacy_bar_timestamps=True,
                )
            ],
            strategy=MultiInstrumentStrategyConfig(
                strategy_file=plugin,
                strategy_params={"path": str(lifecycle_output)},
            ),
            portfolio=MultiInstrumentPortfolioConfig(initial_capital=1000.0),
            execution=MultiInstrumentExecutionConfig(
                trading_days_per_year=252,
                timing="signal_close_to_same_close",
                allow_noncausal_same_close=True,
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "out", run_id="lifecycle"),
        )
    )

    state = json.loads(lifecycle_output.read_text(encoding="utf-8"))
    assert state["calls"] == 2
    assert state["fills"] == 1
    assert state["event_count"] == 2


def test_multi_instrument_max_leverage_can_clip_targets(tmp_path: Path) -> None:
    asset_a = write_intraday_dataset(tmp_path, "asset_a.parquet", "A", [100.0, 101.0])
    asset_b = write_intraday_dataset(tmp_path, "asset_b.parquet", "B", [50.0, 49.0])
    plugin = write_strategy(tmp_path, 'def generate_targets(ctx):\n    return {"A": 1, "B": 1}\n')

    result = run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=[
                MultiInstrumentLegConfig(
                    asset_id="A",
                    dataset_path=asset_a,
                    quality_approval_required=False,
                    allow_legacy_bar_timestamps=True,
                ),
                MultiInstrumentLegConfig(
                    asset_id="B",
                    dataset_path=asset_b,
                    quality_approval_required=False,
                    allow_legacy_bar_timestamps=True,
                ),
            ],
            strategy=MultiInstrumentStrategyConfig(strategy_file=plugin),
            portfolio=MultiInstrumentPortfolioConfig(
                initial_capital=1000.0,
                fixed_contracts=10.0,
                max_leverage=0.1,
                risk_policy="clip",
            ),
            execution=MultiInstrumentExecutionConfig(
                trading_days_per_year=252,
                timing="signal_close_to_same_close",
                allow_noncausal_same_close=True,
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "out", run_id="clip"),
        )
    )

    equity = pl.read_parquet(result.output_dir / "portfolio_equity.parquet")
    assert result.metrics["risk_clipped_count"] == 1
    assert equity.row(0, named=True)["gross_position_notional"] == pytest.approx(100.0)


def test_cli_multi_instrument_run_smoke(tmp_path: Path) -> None:
    asset_a = write_intraday_dataset(tmp_path, "asset_a.parquet", "A", [100.0, 101.0])
    asset_b = write_intraday_dataset(tmp_path, "asset_b.parquet", "B", [50.0, 49.0])
    plugin = write_strategy(tmp_path, 'def generate_targets(ctx):\n    return {"A": 1, "B": -1}\n')

    result = runner.invoke(
        app,
        [
            "multi-instrument-run",
            "--leg",
            f"A={asset_a},quality_approval_required=false,allow_legacy_bar_timestamps=true",
            "--leg",
            f"B={asset_b},quality_approval_required=false,allow_legacy_bar_timestamps=true",
            "--strategy-file",
            str(plugin),
            "--from-date",
            "2024-01-02",
            "--initial-capital",
            "1000",
            "--output-root",
            str(tmp_path / "out"),
            "--run-id",
            "cli-multi",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "asset_count=2" in result.stdout
    assert (tmp_path / "out" / "cli-multi" / "portfolio_equity.parquet").exists()
