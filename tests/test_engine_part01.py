# ruff: noqa
from test_engine_support import *  # noqa: F401,F403


def test_approved_manifest_allows_backtest(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    result = run_backtest(make_config(tmp_path, dataset_path, manifest_path))

    assert result.dataset_quality.approved_for_backtest is True
    assert (result.output_dir / "equity.parquet").exists()
    assert (result.output_dir / "metrics.json").exists()


def test_manifest_entry_for_another_dataset_path_cannot_approve_run(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, adjustment_mode="back_adjusted")
    manifest_path = write_manifest(
        tmp_path,
        dataset_path=tmp_path
        / "continuous_symbol=TEST_CONT"
        / "adjustment_mode=unadjusted"
        / "x.parquet",
        status="approved",
    )

    with pytest.raises(QualityGateError, match="status=missing"):
        run_backtest(make_config(tmp_path, dataset_path, manifest_path))


@pytest.mark.parametrize("status", ["split_required", "rejected"])
def test_blocked_manifest_statuses_fail(tmp_path: Path, status: str) -> None:
    dataset_path = write_dataset(tmp_path)
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status=status)

    with pytest.raises(QualityGateError, match=f"status={status}"):
        run_backtest(make_config(tmp_path, dataset_path, manifest_path))

    assert not (tmp_path / "out" / "run").exists()


def test_missing_manifest_entry_blocks_by_default(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    manifest_path = write_manifest(
        tmp_path,
        dataset_path=tmp_path / "other.parquet",
        status="approved",
        symbol="OTHER_CONT",
    )

    with pytest.raises(QualityGateError, match="status=missing"):
        run_backtest(make_config(tmp_path, dataset_path, manifest_path))


def test_required_column_validation(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    df = pl.read_parquet(dataset_path).drop("close")
    df.write_parquet(dataset_path)

    with pytest.raises(DatasetValidationError, match="missing required columns"):
        validate_dataset(load_dataset(dataset_path), dataset_path)


def test_duplicate_ts_validation(tmp_path: Path) -> None:
    ts = [datetime(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 3)]
    dataset_path = write_dataset(tmp_path, ts=ts)

    with pytest.raises(DatasetValidationError, match="duplicates"):
        validate_dataset(load_dataset(dataset_path), dataset_path)


def test_unsorted_ts_validation(tmp_path: Path) -> None:
    ts = [datetime(2024, 1, 2), datetime(2024, 1, 1), datetime(2024, 1, 3)]
    dataset_path = write_dataset(tmp_path, ts=ts)

    with pytest.raises(DatasetValidationError, match="sorted ascending"):
        validate_dataset(load_dataset(dataset_path), dataset_path)


def test_buy_and_hold_strategy_returns_and_equity(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[100.0, 110.0, 121.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")

    result = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            strategy=StrategyConfig(name="buy_and_hold"),
        )
    )
    equity = pl.read_parquet(result.output_dir / "equity.parquet")

    assert equity.get_column("signal").to_list() == [1.0, 1.0, 1.0]
    assert equity.get_column("position").to_list() == [0.0, 1.0, 1.0]
    assert equity.get_column("return_1d").to_list() == pytest.approx([0.0, 0.1, 0.1])
    assert equity.get_column("equity").to_list() == pytest.approx([1.0, 1.1, 1.21])


def test_moving_average_crossover_strategy(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[1.0, 1.0, 2.0, 3.0, 4.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")

    result = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            strategy=StrategyConfig(
                name="moving_average_crossover",
                fast_window=2,
                slow_window=3,
            ),
        )
    )
    equity = pl.read_parquet(result.output_dir / "equity.parquet")

    assert equity.get_column("signal").to_list() == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert equity.get_column("position").to_list() == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_no_lookahead_timing_convention(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[10.0, 100.0, 1.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")

    result = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            strategy=StrategyConfig(
                name="moving_average_crossover",
                fast_window=1,
                slow_window=2,
            ),
        )
    )
    equity = pl.read_parquet(result.output_dir / "equity.parquet")

    assert equity.get_column("signal").to_list() == [0.0, 1.0, 0.0]
    assert equity.get_column("position").to_list() == [0.0, 0.0, 1.0]
    assert equity.get_column("strategy_return").to_list() == pytest.approx([0.0, 0.0, -0.99])


def test_cost_model_on_turnover(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[100.0, 110.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")

    result = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            strategy=StrategyConfig(name="buy_and_hold"),
            execution=ExecutionConfig(commission_bps=10.0),
        )
    )
    equity = pl.read_parquet(result.output_dir / "equity.parquet")

    assert equity.get_column("turnover").to_list() == [0.0, 2.0]
    assert equity.get_column("cost").to_list() == pytest.approx([0.0, 0.002])
    assert equity.get_column("strategy_return_after_cost").to_list() == pytest.approx([0.0, 0.098])
    assert equity.get_column("equity").to_list() == pytest.approx([1.0, 1.098])


def test_trade_events_generated_on_position_changes(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[10.0, 20.0, 30.0, 10.0, 5.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")

    result = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            strategy=StrategyConfig(
                name="moving_average_crossover",
                fast_window=1,
                slow_window=2,
            ),
        )
    )
    trade_events = pl.read_parquet(result.output_dir / "trade_events.parquet")

    assert trade_events.get_column("event").to_list() == ["enter_long", "exit_long"]
    assert trade_events.get_column("old_position").to_list() == [0.0, 1.0]
    assert trade_events.get_column("new_position").to_list() == [1.0, 0.0]
    assert trade_events.get_column("price").to_list() == [20.0, 10.0]
    assert trade_events.get_column("ts").to_list() == [
        datetime(2024, 1, 2),
        datetime(2024, 1, 4),
    ]


def test_one_long_trade_entry_and_exit(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[100.0, 110.0, 120.0, 130.0, 125.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_signal_plugin(tmp_path, [1.0, 1.0, 1.0, 0.0, 0.0])

    result = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            strategy=StrategyConfig(name="plugin", strategy_file=plugin_path),
        )
    )
    trades = pl.read_parquet(result.output_dir / "trades.parquet")

    assert trades.height == 1
    trade = trades.row(0, named=True)
    assert trade["direction"] == "long"
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["exit_price"] == pytest.approx(130.0)
    assert trade["holding_bars"] == 3
    assert trade["gross_return"] == pytest.approx(0.3)
    assert trade["net_return"] == pytest.approx(0.3)
    assert trade["pnl_contribution"] == pytest.approx(0.3)
    assert trade["winning_trade"] is True
    assert trade["closed_by_end_of_data"] is False


def test_open_trade_closed_at_end_of_data(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[100.0, 110.0, 121.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")

    result = run_backtest(make_config(tmp_path, dataset_path, manifest_path))
    trades = pl.read_parquet(result.output_dir / "trades.parquet")

    assert trades.height == 1
    trade = trades.row(0, named=True)
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["exit_price"] == pytest.approx(121.0)
    assert trade["holding_bars"] == 2
    assert trade["closed_by_end_of_data"] is True


def test_no_trades_returns_null_trade_metrics(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[100.0, 101.0, 102.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_plugin(tmp_path, "def generate_signals(ctx):\n    return ctx.flat()\n")

    result = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            strategy=StrategyConfig(name="plugin", strategy_file=plugin_path),
        )
    )
    trades = pl.read_parquet(result.output_dir / "trades.parquet")

    assert trades.height == 0
    assert result.metrics.trade_count == 0
    assert result.metrics.win_rate is None
    assert result.metrics.avg_win is None
    assert result.metrics.avg_loss is None
    assert result.metrics.profit_factor is None
    assert result.metrics.best_trade is None
    assert result.metrics.worst_trade is None
    assert result.metrics.avg_trade_return is None
    assert result.metrics.avg_holding_bars is None
    assert result.metrics.long_trade_count == 0


def test_trade_metrics_win_loss_and_profit_factor(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[100.0, 110.0, 100.0, 90.0, 80.0, 80.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_signal_plugin(tmp_path, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    result = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            strategy=StrategyConfig(name="plugin", strategy_file=plugin_path),
        )
    )

    assert result.metrics.trade_count == 2
    assert result.metrics.win_rate == pytest.approx(0.5)
    assert result.metrics.avg_win == pytest.approx(0.1)
    assert result.metrics.avg_loss == pytest.approx(80.0 / 90.0 - 1.0)
    assert result.metrics.profit_factor == pytest.approx(0.9)
    assert result.metrics.best_trade == pytest.approx(0.1)
    assert result.metrics.worst_trade == pytest.approx(80.0 / 90.0 - 1.0)
    assert result.metrics.avg_trade_return == pytest.approx((0.1 + (80.0 / 90.0 - 1.0)) / 2)
    assert result.metrics.avg_holding_bars == pytest.approx(1.0)
    assert result.metrics.long_trade_count == 2
    assert result.metrics.closed_by_end_of_data_count == 0
