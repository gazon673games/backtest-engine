# ruff: noqa
from test_engine_support import *  # noqa: F401,F403


def test_mfe_mae_for_simple_price_path(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[100.0, 120.0, 80.0, 110.0, 105.0])
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
    trade = pl.read_parquet(result.output_dir / "trades.parquet").row(0, named=True)

    assert trade["gross_return"] == pytest.approx(0.1)
    assert trade["max_favorable_excursion"] == pytest.approx(0.2)
    assert trade["max_adverse_excursion"] == pytest.approx(-0.2)


def test_trade_cost_includes_entry_and_exit_costs(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[100.0, 110.0, 120.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_signal_plugin(tmp_path, [1.0, 0.0, 0.0])

    result = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            strategy=StrategyConfig(name="plugin", strategy_file=plugin_path),
            execution=ExecutionConfig(commission_bps=10.0),
        )
    )
    trade = pl.read_parquet(result.output_dir / "trades.parquet").row(0, named=True)

    assert trade["gross_return"] == pytest.approx(0.1)
    assert trade["cost"] == pytest.approx(0.002)
    assert trade["net_return"] == pytest.approx((1.0 + 0.099) * (1.0 - 0.001) - 1.0)
    assert trade["pnl_contribution"] == pytest.approx(trade["net_return"])


def test_metrics_calculation(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[100.0, 110.0, 121.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")

    result = run_backtest(make_config(tmp_path, dataset_path, manifest_path))

    assert result.metrics.bar_count == 3
    assert result.metrics.total_return == pytest.approx(0.21)
    assert result.metrics.trade_count == 1
    assert result.metrics.exposure == pytest.approx(1.0)
    assert result.metrics.win_rate == pytest.approx(1.0)
    assert result.metrics.avg_win == pytest.approx(0.21)
    assert result.metrics.avg_loss is None
    assert result.metrics.best_trade == pytest.approx(0.21)
    assert result.metrics.worst_trade == pytest.approx(0.21)


def test_annualization_uses_return_intervals_not_price_rows(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[100.0, 110.0, 121.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")

    result = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            execution=ExecutionConfig(trading_days_per_year=2),
        )
    )

    assert result.metrics.annualized_return == pytest.approx(0.21)


def test_cli_inspect_dataset(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")

    result = runner.invoke(
        app,
        [
            "inspect-dataset",
            "--dataset-path",
            str(dataset_path),
            "--quality-manifest",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 0
    assert "continuous_symbol=TEST_CONT" in result.stdout
    assert "quality_gate_status=approved" in result.stdout
    assert "approved_for_backtest=true" in result.stdout


def test_cli_run_smoke_test(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")

    result = runner.invoke(
        app,
        [
            "run",
            "--dataset-path",
            str(dataset_path),
            "--quality-manifest",
            str(manifest_path),
            "--strategy",
            "moving_average_crossover",
            "--fast-window",
            "2",
            "--slow-window",
            "3",
            "--output-root",
            str(tmp_path / "cli-out"),
            "--run-id",
            "cli-run",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "cli-out" / "cli-run" / "equity.parquet").exists()
    assert "status=approved" in result.stdout


def test_plugin_file_with_generate_signals_loads(tmp_path: Path) -> None:
    plugin_path = write_plugin(
        tmp_path,
        "def generate_signals(ctx):\n    return ctx.flat()\n",
    )

    assert callable(load_strategy_plugin(plugin_path))


def test_missing_plugin_file_fails(tmp_path: Path) -> None:
    with pytest.raises(StrategyPluginError, match="does not exist"):
        load_strategy_plugin(tmp_path / "missing.py")


def test_non_py_plugin_file_fails(tmp_path: Path) -> None:
    plugin_path = tmp_path / "strategy.txt"
    plugin_path.write_text("def generate_signals(ctx):\n    return ctx.flat()\n", encoding="utf-8")

    with pytest.raises(StrategyPluginError, match=r"\.py file"):
        load_strategy_plugin(plugin_path)


def test_plugin_without_generate_signals_fails(tmp_path: Path) -> None:
    plugin_path = write_plugin(tmp_path, "VALUE = 1\n")

    with pytest.raises(StrategyPluginError, match="must define generate_signals"):
        load_strategy_plugin(plugin_path)


def test_plugin_returning_wrong_length_fails(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_plugin(
        tmp_path,
        "def generate_signals(ctx):\n    return [0.0, 1.0]\n",
    )

    with pytest.raises(StrategyPluginError, match="wrong length"):
        run_backtest(
            make_config(
                tmp_path,
                dataset_path,
                manifest_path,
                strategy=StrategyConfig(name="plugin", strategy_file=plugin_path),
            )
        )


def test_plugin_returning_invalid_values_fails(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[1.0, 2.0, 3.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_plugin(
        tmp_path,
        "def generate_signals(ctx):\n    return [0.0, -1.0, 1.0]\n",
    )

    with pytest.raises(StrategyPluginError, match="unsupported signal value"):
        run_backtest(
            make_config(
                tmp_path,
                dataset_path,
                manifest_path,
                strategy=StrategyConfig(name="plugin", strategy_file=plugin_path),
            )
        )


def test_plugin_ma_output_matches_builtin_ma_output(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[1.0, 1.0, 2.0, 3.0, 4.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_plugin(
        tmp_path,
        """
def generate_signals(ctx, params=None):
    params = params or {}
    fast = ctx.sma("close", int(params.get("fast_window", 2)))
    slow = ctx.sma("close", int(params.get("slow_window", 3)))
    return ctx.long_when(fast > slow)
""".lstrip(),
    )

    built_in = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            strategy=StrategyConfig(
                name="moving_average_crossover",
                fast_window=2,
                slow_window=3,
            ),
            run_id="builtin",
        )
    )
    plugin = run_backtest(
        make_config(
            tmp_path,
            dataset_path,
            manifest_path,
            strategy=StrategyConfig(
                name="plugin",
                strategy_file=plugin_path,
                strategy_params={"fast_window": "2", "slow_window": "3"},
            ),
            run_id="plugin",
        )
    )

    built_in_equity = pl.read_parquet(built_in.output_dir / "equity.parquet")
    plugin_equity = pl.read_parquet(plugin.output_dir / "equity.parquet")
    assert (
        plugin_equity.get_column("signal").to_list()
        == built_in_equity.get_column("signal").to_list()
    )
    assert (
        plugin_equity.get_column("position").to_list()
        == built_in_equity.get_column("position").to_list()
    )


def test_cli_run_script_writes_outputs_and_config_hash(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_plugin(
        tmp_path,
        (
            "def generate_signals(ctx, params=None):\n"
            "    return ctx.long_when(ctx.series('close') > 0)\n"
        ),
    )

    result = runner.invoke(
        app,
        [
            "run-script",
            "--dataset-path",
            str(dataset_path),
            "--quality-manifest",
            str(manifest_path),
            "--strategy-file",
            str(plugin_path),
            "--param",
            "example=true",
            "--output-root",
            str(tmp_path / "script-out"),
            "--run-id",
            "plugin-run",
            "--write-csv",
        ],
    )

    output_dir = tmp_path / "script-out" / "plugin-run"
    assert result.exit_code == 0
    assert (output_dir / "config.json").exists()
    assert (output_dir / "dataset_quality.json").exists()
    assert (output_dir / "equity.parquet").exists()
    assert (output_dir / "equity.csv").exists()
    assert (output_dir / "trades.parquet").exists()
    assert (output_dir / "trades.csv").exists()
    assert (output_dir / "trade_events.parquet").exists()
    assert (output_dir / "trade_events.csv").exists()
    assert (output_dir / "metrics.json").exists()

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["strategy_type"] == "plugin"
    assert config["strategy_file"] == str(plugin_path)
    assert config["strategy_file_sha256"] == strategy_file_sha256(plugin_path)
    assert config["strategy_params"] == {"example": "true"}
