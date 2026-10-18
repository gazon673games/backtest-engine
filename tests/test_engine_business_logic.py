# ruff: noqa
from test_engine_support import *  # noqa: F401,F403


def test_manifest_cannot_approve_replaced_file_at_same_path(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    pl.read_parquet(dataset_path).with_columns(
        (pl.col("volume") + 1.0).alias("volume")
    ).write_parquet(dataset_path)

    with pytest.raises(QualityGateError, match="status=mismatch"):
        run_backtest(make_config(tmp_path, dataset_path, manifest_path))


def test_roll_charges_close_and_reopen_turnover(tmp_path: Path) -> None:
    dataset_path = write_dataset(
        tmp_path,
        closes=[100.0, 101.0, 102.0],
        active_contracts=["TESTH4", "TESTM4", "TESTM4"],
    )
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    result = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            execution=ExecutionConfig(commission_bps=10.0),
        )
    )
    equity = pl.read_parquet(result.output_dir / "equity.parquet")
    events = pl.read_parquet(result.output_dir / "trade_events.parquet")

    assert equity.get_column("position_turnover").to_list() == [0.0, 1.0, 1.0]
    assert equity.get_column("end_of_data_turnover").to_list() == [0.0, 0.0, 1.0]
    assert equity.get_column("roll_turnover").to_list() == [0.0, 2.0, 0.0]
    assert equity.get_column("turnover").to_list() == [0.0, 3.0, 1.0]
    assert equity.get_column("cost").to_list() == pytest.approx([0.0, 0.003, 0.001])
    assert events.get_column("event").to_list() == [
        "enter_long",
        "roll_long",
        "exit_long_end_of_data",
    ]
    assert events.row(1, named=True)["previous_active_contract"] == "TESTH4"
    assert events.row(1, named=True)["active_contract"] == "TESTM4"


def test_unadjusted_roll_gap_cannot_be_counted_as_daily_pnl(tmp_path: Path) -> None:
    dataset_path = write_dataset(
        tmp_path,
        adjustment_mode="unadjusted",
        closes=[100.0, 200.0],
        active_contracts=["TESTH4", "TESTM4"],
    )
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")

    with pytest.raises(DatasetValidationError, match="unadjusted contract gaps"):
        run_backtest(make_config(tmp_path, dataset_path, manifest_path))


def test_unadjusted_roll_is_allowed_while_strategy_is_flat(tmp_path: Path) -> None:
    dataset_path = write_dataset(
        tmp_path,
        adjustment_mode="unadjusted",
        closes=[100.0, 200.0],
        active_contracts=["TESTH4", "TESTM4"],
    )
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_signal_plugin(tmp_path, [0.0, 0.0])

    result = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            strategy=StrategyConfig(name="plugin", strategy_file=plugin_path),
        )
    )

    assert result.metrics.total_return == 0.0


def test_trade_pnl_contributions_reconcile_to_equity_curve(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[100.0, 110.0, 110.0, 110.0, 121.0, 121.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_signal_plugin(tmp_path, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    result = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            strategy=StrategyConfig(name="plugin", strategy_file=plugin_path),
            execution=ExecutionConfig(commission_bps=10.0),
        )
    )
    trades = pl.read_parquet(result.output_dir / "trades.parquet")

    assert trades.height == 2
    assert trades.get_column("pnl_contribution").sum() == pytest.approx(result.metrics.total_return)
