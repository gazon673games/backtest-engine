# ruff: noqa: F405
from test_engine_support import *  # noqa: F401,F403


def test_plugin_quality_gate_blocks_split_required_dataset(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="split_required")
    plugin_path = write_plugin(
        tmp_path,
        "def generate_signals(ctx):\n    return ctx.long_when(ctx.series('close') > 0)\n",
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
            "--output-root",
            str(tmp_path / "blocked-out"),
            "--run-id",
            "blocked",
        ],
    )

    assert result.exit_code == 1
    assert "status=split_required" in result.output
    assert not (tmp_path / "blocked-out" / "blocked").exists()


def test_batch_grid_expansion() -> None:
    params = expand_grid({"fast_window": [5, 10], "slow_window": [20, 60]})

    assert params == [
        {"fast_window": 5, "slow_window": 20},
        {"fast_window": 5, "slow_window": 60},
        {"fast_window": 10, "slow_window": 20},
        {"fast_window": 10, "slow_window": 60},
    ]


def test_batch_fast_window_lt_slow_window_constraint() -> None:
    params = expand_grid(
        {"fast_window": [20, 60], "slow_window": [20, 60]},
        ["fast_window_lt_slow_window"],
    )

    assert params == [{"fast_window": 20, "slow_window": 60}]


def test_batch_deterministic_run_id_generation(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[1.0, 2.0, 3.0])
    strategy_file = write_ma_plugin(tmp_path)
    run_id = generate_run_id(
        continuous_symbol="CNYRUB_CONT",
        adjustment_mode="back_adjusted",
        strategy_file=strategy_file,
        dataset_path=dataset_path,
        execution_params={},
        params={"slow_window": 60, "fast_window": 20},
    )

    assert run_id.startswith("CNYRUB_CONT_back_adjusted_")
    assert len(run_id.rsplit("_", 1)[-1]) == 16


def test_batch_run_two_parameter_combinations(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[1.0, 2.0, 3.0, 4.0, 5.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_ma_plugin(tmp_path)

    result = run_batch(
        BatchConfig(
            experiment_id="grid_test",
            output_root=tmp_path / "batch",
            dataset_paths=[dataset_path],
            quality_manifest=manifest_path,
            strategy_file=plugin_path,
            grid={"fast_window": [1], "slow_window": [2, 3]},
            constraints=["fast_window_lt_slow_window"],
        )
    )

    assert result.successful_run_count == 2
    assert result.failed_run_count == 0
    summary = pl.read_csv(result.output_dir / "summary.csv")
    assert summary.height == 2
    assert sorted(summary.get_column("slow_window").to_list()) == [2, 3]
    assert all((Path(path) / "metrics.json").exists() for path in summary["run_path"])


def test_batch_summary_output_contains_metrics_and_params(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[1.0, 2.0, 3.0, 4.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_ma_plugin(tmp_path)

    result = run_batch(
        BatchConfig(
            trading_days_per_year=252,
            experiment_id="summary_test",
            output_root=tmp_path / "batch",
            dataset_paths=[dataset_path],
            quality_manifest=manifest_path,
            strategy_file=plugin_path,
            grid={"fast_window": [1], "slow_window": [2]},
            constraints=["fast_window_lt_slow_window"],
        )
    )
    summary = pl.read_parquet(result.output_dir / "summary.parquet")

    assert summary.height == 1
    row = summary.row(0, named=True)
    assert row["fast_window"] == 1
    assert row["slow_window"] == 2
    assert row["status"] == "success"
    assert row["total_return"] is not None
    assert row["sharpe"] is not None


def test_batch_failed_run_output_captures_invalid_params(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[1.0, 2.0, 3.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_ma_plugin(tmp_path)

    result = run_batch(
        BatchConfig(
            experiment_id="failed_params",
            output_root=tmp_path / "batch",
            dataset_paths=[dataset_path],
            quality_manifest=manifest_path,
            strategy_file=plugin_path,
            grid={"fast_window": [0], "slow_window": [2]},
            constraints=["fast_window_lt_slow_window"],
        )
    )
    failed = pl.read_csv(result.output_dir / "failed_runs.csv")

    assert result.successful_run_count == 0
    assert result.failed_run_count == 1
    assert failed.get_column("status").to_list() == ["failed"]
    assert "fast_window" in failed.get_column("params")[0]


def test_batch_split_required_dataset_is_recorded_as_failed(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[1.0, 2.0, 3.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="split_required")
    plugin_path = write_ma_plugin(tmp_path)

    result = run_batch(
        BatchConfig(
            experiment_id="blocked",
            output_root=tmp_path / "batch",
            dataset_paths=[dataset_path],
            quality_manifest=manifest_path,
            strategy_file=plugin_path,
            grid={"fast_window": [1], "slow_window": [2]},
            constraints=["fast_window_lt_slow_window"],
        )
    )
    failed = pl.read_csv(result.output_dir / "failed_runs.csv")

    assert result.successful_run_count == 0
    assert result.failed_run_count == 1
    assert failed.get_column("status").to_list() == ["blocked_by_quality_gate"]
    assert not any((result.output_dir / "runs").glob("*"))


def test_batch_fail_fast_stops_after_first_failure(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[1.0, 2.0, 3.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_ma_plugin(tmp_path)

    result = run_batch(
        BatchConfig(
            experiment_id="fail_fast",
            output_root=tmp_path / "batch",
            dataset_paths=[dataset_path],
            quality_manifest=manifest_path,
            strategy_file=plugin_path,
            grid={"fast_window": [0, 1], "slow_window": [2]},
            constraints=["fast_window_lt_slow_window"],
            continue_on_error=False,
        )
    )

    assert result.successful_run_count == 0
    assert result.failed_run_count == 1


def test_cli_batch_run_smoke_test(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path, closes=[1.0, 2.0, 3.0, 4.0])
    manifest_path = write_manifest(tmp_path, dataset_path=dataset_path, status="approved")
    plugin_path = write_ma_plugin(tmp_path)

    result = runner.invoke(
        app,
        [
            "batch-run",
            "--dataset-path",
            str(dataset_path),
            "--quality-manifest",
            str(manifest_path),
            "--strategy-file",
            str(plugin_path),
            "--grid",
            "fast_window=1",
            "--grid",
            "slow_window=2,3",
            "--constraint",
            "fast_window_lt_slow_window",
            "--output-root",
            str(tmp_path / "batch"),
            "--experiment-id",
            "cli_batch",
        ],
    )

    assert result.exit_code == 0
    assert "successful_run_count=2" in result.stdout
    assert (tmp_path / "batch" / "cli_batch" / "summary.csv").exists()


def test_report_loads_summary_parquet(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path)

    result = generate_experiment_report(ReportConfig(experiment_path=experiment_path, top=2))

    assert result.top_run_count == 2
    assert result.best_overall is not None
    assert result.best_overall["run_id"] == "cny_fast"


def test_report_falls_back_to_summary_csv(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path)
    (experiment_path / "summary.parquet").unlink()

    result = generate_experiment_report(ReportConfig(experiment_path=experiment_path, top=1))

    assert result.top_run_count == 1
    top_runs = pl.read_csv(experiment_path / "top_runs.csv")
    assert top_runs.get_column("run_id").to_list() == ["cny_fast"]


def test_report_sort_by_column_validation(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path)

    with pytest.raises(Exception, match="sort_by column does not exist"):
        generate_experiment_report(
            ReportConfig(experiment_path=experiment_path, sort_by="missing_metric")
        )


def test_report_top_n_selection(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path)

    generate_experiment_report(ReportConfig(experiment_path=experiment_path, top=2))
    top_runs = pl.read_parquet(experiment_path / "top_runs.parquet")

    assert top_runs.height == 2
    assert top_runs.get_column("rank").to_list() == [1, 2]
    assert top_runs.get_column("run_id").to_list() == ["cny_fast", "usd_fast"]
