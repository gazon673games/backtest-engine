# ruff: noqa
from test_intraday_event_workflow_support import *  # noqa: F401,F403


def test_intraday_event_batch_grid_expansion_supports_numbers_and_strings() -> None:
    grid = parse_intraday_event_grid_options(
        ["window_minutes=180,210", "decision_time=13:00,13:30", "commission_bps=0,2.5"]
    )

    assert grid == {
        "window_minutes": [180, 210],
        "decision_time": ["13:00", "13:30"],
        "commission_bps": [0, 2.5],
    }
    assert len(expand_intraday_event_grid(grid)) == 8


def test_intraday_event_batch_splits_engine_and_plugin_params() -> None:
    engine_params, plugin_params = split_intraday_event_params(
        {
            "schedule": "times",
            "decision_time": "13:30",
            "missing_decision_policy": "close",
            "commission_bps": 2,
            "bid_ask_spread_ticks": 2,
            "volume_participation_limit": 0.25,
            "partial_fill_policy": "reject",
            "window_minutes": 210,
            "custom_mode": "long",
        }
    )

    assert engine_params == {
        "schedule": "times",
        "decision_time": "13:30",
        "missing_decision_policy": "close",
        "commission_bps": 2,
        "bid_ask_spread_ticks": 2,
        "volume_participation_limit": 0.25,
        "partial_fill_policy": "reject",
    }
    assert plugin_params == {"window_minutes": 210, "custom_mode": "long"}


def test_intraday_event_batch_deterministic_run_id_generation(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    strategy_file = write_mode_plugin(tmp_path)
    run_id = generate_intraday_event_run_id(
        dataset_path=dataset_path,
        experiment_id="grid",
        dataset_validation={
            "continuous_symbol": "CNYRUB_NEAREST_5M",
            "timeframe": "5m",
        },
        strategy_file=strategy_file,
        engine_params={
            "schedule": "times",
            "decision_time": "13:30",
            "commission_bps": 1,
            "slippage_bps": 0,
        },
        plugin_params={"window_minutes": 210},
    )

    assert run_id.startswith("grid_CNYRUB_NEAREST_5M_5m_")
    assert len(run_id.rsplit("_", 1)[-1]) == 16


def test_intraday_event_batch_routes_engine_and_plugin_params(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_mode_plugin(tmp_path)

    result = run_intraday_event_batch(
        IntradayEventBatchConfig(
            experiment_id="route_test",
            output_root=tmp_path / "batch",
            dataset_paths=[dataset_path],
            quality_approval_required=False,
            allow_legacy_bar_timestamps=True,
            strategy_file=plugin_path,
            timing="signal_close_to_same_close",
            allow_noncausal_same_close=True,
            end_of_data_position_policy="mark",
            schedule="times",
            grid={
                "decision_time": ["13:30"],
                "mode": ["long", "short"],
                "window_minutes": [210],
                "commission_bps": [2],
            },
        )
    )

    summary = pl.read_parquet(result.output_dir / "summary.parquet").sort("mode")

    assert result.successful_run_count == 2
    assert result.failed_run_count == 0
    assert summary.get_column("schedule").to_list() == ["times", "times"]
    assert summary.get_column("decision_time").to_list() == ["13:30", "13:30"]
    assert summary.get_column("missing_decision_policy").to_list() == ["carry", "carry"]
    assert summary.get_column("commission_bps").to_list() == [2.0, 2.0]
    assert summary.get_column("window_minutes").to_list() == ["210", "210"]
    assert summary.get_column("mode").to_list() == ["long", "short"]
    assert summary.get_column("strategy_call_count").to_list() == [1, 1]


def test_intraday_event_batch_summary_files_written(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_mode_plugin(tmp_path)

    result = run_intraday_event_batch(
        IntradayEventBatchConfig(
            experiment_id="summary_test",
            output_root=tmp_path / "batch",
            dataset_paths=[dataset_path],
            quality_approval_required=False,
            allow_legacy_bar_timestamps=True,
            strategy_file=plugin_path,
            timing="signal_close_to_same_close",
            allow_noncausal_same_close=True,
            end_of_data_position_policy="mark",
            schedule="times",
            grid={"decision_time": ["13:30"], "mode": ["long"], "window_minutes": [210]},
        )
    )

    row = pl.read_parquet(result.output_dir / "summary.parquet").row(0, named=True)
    run_path = Path(str(row["run_path"]))

    assert row["experiment_id"] == "summary_test"
    assert row["status"] == "success"
    assert row["continuous_symbol"] == "CNYRUB_NEAREST_5M"
    assert row["timeframe"] == "5m"
    assert row["event_count"] == 2
    assert "sortino" in row
    assert "value_at_risk_95" in row
    assert "liquidation_count" in row
    assert (result.output_dir / "summary.csv").exists()
    assert (result.output_dir / "summary.json").exists()
    assert (result.output_dir / "failed_runs.parquet").exists()
    assert (run_path / "events.parquet").exists()
    assert (run_path / "metrics.json").exists()


def test_intraday_event_batch_failed_run_is_captured(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_mode_plugin(tmp_path)

    result = run_intraday_event_batch(
        IntradayEventBatchConfig(
            experiment_id="failed_test",
            output_root=tmp_path / "batch",
            dataset_paths=[dataset_path],
            quality_approval_required=False,
            allow_legacy_bar_timestamps=True,
            strategy_file=plugin_path,
            end_of_data_position_policy="mark",
            grid={"mode": ["fail", "long"]},
        )
    )
    failed = pl.read_csv(result.output_dir / "failed_runs.csv")

    assert result.successful_run_count == 1
    assert result.failed_run_count == 1
    assert failed.get_column("status").to_list() == ["failed"]
    assert failed.get_column("error_type").to_list() == ["StrategyPluginError"]
    assert "mode fail requested" in failed.get_column("error_message")[0]


def test_intraday_event_batch_fail_fast_stops_after_first_failure(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_mode_plugin(tmp_path)

    result = run_intraday_event_batch(
        IntradayEventBatchConfig(
            experiment_id="fail_fast_test",
            output_root=tmp_path / "batch",
            dataset_paths=[dataset_path],
            quality_approval_required=False,
            allow_legacy_bar_timestamps=True,
            strategy_file=plugin_path,
            grid={"mode": ["fail", "long"]},
            continue_on_error=False,
        )
    )

    assert result.successful_run_count == 0
    assert result.failed_run_count == 1


def test_cli_intraday_event_batch_run_smoke(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_mode_plugin(tmp_path)

    result = runner.invoke(
        app,
        [
            "intraday-event-batch-run",
            "--dataset-path",
            str(dataset_path),
            "--strategy-file",
            str(plugin_path),
            "--no-require-quality-approval",
            "--allow-legacy-bar-timestamps",
            "--from-date",
            "2024-01-02",
            "--schedule",
            "times",
            "--execution-timing",
            "signal_close_to_same_close",
            "--allow-noncausal-same-close",
            "--end-of-data-position-policy",
            "mark",
            "--grid",
            "decision_time=13:30",
            "--grid",
            "mode=long,flat",
            "--grid",
            "commission_bps=0,1",
            "--output-root",
            str(tmp_path / "batch"),
            "--experiment-id",
            "cli_event_batch",
        ],
    )

    output_dir = tmp_path / "batch" / "cli_event_batch"
    assert result.exit_code == 0, result.stdout
    assert "successful_run_count=4" in result.stdout
    assert "failed_run_count=0" in result.stdout
    assert (output_dir / "summary.parquet").exists()
    assert (output_dir / "failed_runs.csv").exists()


def test_intraday_event_batch_loads_toml_config(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_mode_plugin(tmp_path)
    config_path = tmp_path / "event_batch.toml"
    config_path.write_text(
        f"""
experiment_id = "toml_event_batch"
output_root = "{(tmp_path / "batch").as_posix()}"
dataset_paths = ["{dataset_path.as_posix()}"]
strategy_file = "{plugin_path.as_posix()}"
quality_approval_required = false
allow_legacy_bar_timestamps = true
from_date = "2024-01-02"
schedule = "times"
timing = "signal_close_to_next_bar_open_to_close"
decision_tolerance_minutes = 5
missing_decision_policy = "close"
bid_ask_spread_ticks = 2.0
volume_participation_limit = 0.25
partial_fill_policy = "reject"
intrabar_ambiguity_policy = "optimistic"
max_workers = 2

[instrument]
contract_multiplier = 10.0
lot_size = 2.0

[portfolio]
initial_capital = 10000.0
sizing_mode = "notional"
target_notional = 5000.0

[session]
session_filter = "main"
main_session_start = "10:00"
main_session_end = "18:50"

[grid]
decision_time = ["13:30"]
mode = ["long"]
""".lstrip(),
        encoding="utf-8",
    )

    config = load_intraday_event_batch_config(config_path)

    assert config.experiment_id == "toml_event_batch"
    assert config.schedule == "times"
    assert config.timing == "signal_close_to_next_bar_open_to_close"
    assert config.missing_decision_policy == "close"
    assert config.bid_ask_spread_ticks == 2.0
    assert config.volume_participation_limit == 0.25
    assert config.partial_fill_policy == "reject"
    assert config.intrabar_ambiguity_policy == "optimistic"
    assert config.max_workers == 2
    assert config.instrument.contract_multiplier == 10.0
    assert config.instrument.lot_size == 2.0
    assert config.portfolio.sizing_mode == "notional"
    assert config.portfolio.target_notional == 5000.0
    assert config.session.session_filter == "main"
    assert config.grid == {"decision_time": ["13:30"], "mode": ["long"]}
