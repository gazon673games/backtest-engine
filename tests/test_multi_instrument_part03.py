# ruff: noqa
import json

from test_multi_instrument_support import *  # noqa: F401,F403


def test_multi_instrument_uses_catalog_quality_registry_and_calendar(tmp_path: Path) -> None:
    dataset = write_intraday_dataset(tmp_path, "asset_a.parquet", "A", [100.0, 101.0])
    catalog = write_catalog(
        tmp_path,
        dataset,
        registry_payload={
            "instruments": [
                {
                    "symbol": "A",
                    "contract_multiplier": 10.0,
                    "lot_size": 1.0,
                    "tick_size": 1.0,
                    "tick_value": 10.0,
                    "initial_margin": 100.0,
                }
            ]
        },
    )
    plugin = write_strategy(tmp_path, 'def generate_targets(ctx):\n    return {"A": 1}\n')

    result = run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=[
                MultiInstrumentLegConfig(
                    asset_id="A",
                    catalog_path=catalog,
                    dataset_id="A",
                    catalog_enforcement="strict",
                    quality_approval_required=True,
                    allow_legacy_bar_timestamps=True,
                )
            ],
            strategy=MultiInstrumentStrategyConfig(strategy_file=plugin),
            portfolio=MultiInstrumentPortfolioConfig(initial_capital=1000.0),
            execution=MultiInstrumentExecutionConfig(
                timing="signal_close_to_same_close",
                allow_noncausal_same_close=True,
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "out", run_id="catalog"),
        )
    )

    legs = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    ledger = pl.read_parquet(result.output_dir / "portfolio_ledger.parquet")
    manifest = json.loads((result.output_dir / "experiment_manifest.json").read_text())

    assert result.metrics["final_equity_cash"] == pytest.approx(1010.0)
    assert legs.row(0, named=True)["contract_multiplier"] == pytest.approx(10.0)
    assert legs.row(0, named=True)["margin_requirement"] == pytest.approx(100.0)
    assert ledger["equity_cash"][-1] == pytest.approx(1010.0)
    assert not (result.output_dir / "forts_accounting.parquet").exists()
    assert manifest["datasets"]["A"]["catalog"]["status"] == "approved"
    assert manifest["datasets"]["A"]["quality_approval"]["status"] == "approved"
    assert manifest["outputs"]["portfolio_ledger.parquet"]
