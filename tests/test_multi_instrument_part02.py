# ruff: noqa
from test_multi_instrument_support import *  # noqa: F401,F403


def test_multi_instrument_execution_timing_can_use_next_open_to_close(tmp_path: Path) -> None:
    dataset = write_ohlc_dataset(
        tmp_path,
        "asset_a.parquet",
        "A",
        opens=[100.0, 102.0, 105.0],
        highs=[100.0, 106.0, 105.0],
        lows=[100.0, 101.0, 105.0],
        closes=[100.0, 105.0, 105.0],
    )
    plugin = write_strategy(tmp_path, 'def generate_targets(ctx):\n    return {"A": 1}\n')

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
                timing="signal_close_to_next_bar_open_to_close",
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "out", run_id="timing"),
        )
    )

    first = pl.read_parquet(result.output_dir / "portfolio_legs.parquet").row(0, named=True)
    assert first["fill_price"] == pytest.approx(102.0)
    assert first["exit_price"] == pytest.approx(105.0)
    assert first["gross_pnl"] == pytest.approx(3.0)


def test_multi_instrument_margin_usage_can_clip_targets(tmp_path: Path) -> None:
    dataset = write_intraday_dataset(tmp_path, "asset_a.parquet", "A", [100.0, 101.0])
    plugin = write_strategy(tmp_path, 'def generate_targets(ctx):\n    return {"A": 1}\n')

    result = run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=[
                MultiInstrumentLegConfig(
                    asset_id="A",
                    dataset_path=dataset,
                    initial_margin=600.0,
                    quality_approval_required=False,
                    allow_legacy_bar_timestamps=True,
                )
            ],
            strategy=MultiInstrumentStrategyConfig(strategy_file=plugin),
            portfolio=MultiInstrumentPortfolioConfig(
                initial_capital=1000.0,
                fixed_contracts=2.0,
                max_margin_usage=0.5,
                risk_policy="clip",
            ),
            execution=MultiInstrumentExecutionConfig(
                timing="signal_close_to_same_close",
                allow_noncausal_same_close=True,
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "out", run_id="margin"),
        )
    )

    legs = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    assert result.metrics["risk_clipped_count"] == 1
    assert legs.row(0, named=True)["target_contracts"] == pytest.approx(5.0 / 6.0)


def test_data_lifecycle_audit_approves_strict_catalog_entry(tmp_path: Path) -> None:
    dataset = write_intraday_dataset(tmp_path, "asset_a.parquet", "A", [100.0, 101.0])
    catalog = write_catalog(tmp_path, dataset)

    result = audit_data_lifecycle(
        DataLifecycleAuditConfig(catalog_path=catalog, output_path=tmp_path / "audit")
    )
    audit = pl.read_parquet(result.audit_path.with_suffix(".parquet"))

    assert result.approved_count == 1
    assert result.rejected_count == 0
    assert audit.row(0, named=True)["status"] == "approved"
    assert "Data Lifecycle Audit" in result.report_path.read_text(encoding="utf-8")


def test_multi_instrument_roll_gap_is_not_counted_as_pnl(tmp_path: Path) -> None:
    dataset = write_intraday_dataset(tmp_path, "asset_a.parquet", "A", [100.0, 200.0, 210.0])
    pl.read_parquet(dataset).with_columns(
        [
            pl.Series("active_contract", ["AH4", "AM4", "AM4"]),
            pl.Series("source_contract_symbol", ["AH4", "AM4", "AM4"]),
        ]
    ).write_parquet(dataset)
    plugin = write_strategy(tmp_path, 'def generate_targets(ctx):\n    return {"A": 1}\n')

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
                timing="signal_close_to_same_close",
                allow_noncausal_same_close=True,
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
                roll_pnl_policy="exclude_gap",
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "out", run_id="roll"),
        )
    )
    legs = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")

    assert legs.row(0, named=True)["contract_roll"] is True
    assert legs.row(0, named=True)["gross_pnl"] == 0.0
    assert result.metrics["gross_pnl"] == pytest.approx(10.0)


def test_data_lifecycle_audit_rejects_broken_registry(tmp_path: Path) -> None:
    dataset = write_intraday_dataset(tmp_path, "asset_a.parquet", "A", [100.0, 101.0])
    catalog = write_catalog(tmp_path, dataset, broken_registry=True)

    result = audit_data_lifecycle(
        DataLifecycleAuditConfig(catalog_path=catalog, output_path=tmp_path / "audit")
    )
    audit = pl.read_parquet(result.audit_path.with_suffix(".parquet"))

    assert result.approved_count == 0
    assert result.rejected_count == 1
    assert audit.row(0, named=True)["instrument_registry_status"] == "rejected"
    assert "contract_multiplier" in audit.row(0, named=True)["violations"]
