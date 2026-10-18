# ruff: noqa
from test_intraday_event_workflow_support import *  # noqa: F401,F403


def test_cli_intraday_event_batch_run_accepts_toml_config(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_mode_plugin(tmp_path)
    config_path = tmp_path / "event_batch.toml"
    config_path.write_text(
        f"""
experiment_id = "cli_event_batch_toml"
output_root = "{(tmp_path / "batch").as_posix()}"
dataset_paths = ["{dataset_path.as_posix()}"]
strategy_file = "{plugin_path.as_posix()}"
quality_approval_required = false
allow_legacy_bar_timestamps = true
from_date = "2024-01-02"
schedule = "times"

[grid]
decision_time = ["13:30"]
mode = ["long", "flat"]
""".lstrip(),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["intraday-event-batch-run", "--config", str(config_path)])

    output_dir = tmp_path / "batch" / "cli_event_batch_toml"
    assert result.exit_code == 0, result.stdout
    assert "successful_run_count=2" in result.stdout
    assert (output_dir / "summary.parquet").exists()


def test_intraday_event_report_reads_parquet_and_writes_outputs(tmp_path: Path) -> None:
    experiment_dir = make_report_experiment(tmp_path)

    result = generate_intraday_event_report(
        IntradayEventReportConfig(experiment_path=experiment_dir, top=3, charts=True)
    )

    assert result.top_run_count == 3
    assert result.failed_run_count == 0
    assert (experiment_dir / "report.md").exists()
    assert (experiment_dir / "top_runs.csv").exists()
    assert (experiment_dir / "top_runs.parquet").exists()
    assert (experiment_dir / "top_runs.json").exists()
    assert (experiment_dir / "best_by_schedule.csv").exists()
    assert (experiment_dir / "best_by_cost.csv").exists()
    assert (experiment_dir / "charts" / "best_overall_equity.png").exists()
    assert (experiment_dir / "charts" / "chart_manifest.json").exists()


def test_intraday_event_report_falls_back_to_csv(tmp_path: Path) -> None:
    experiment_dir = make_report_experiment(tmp_path)
    (experiment_dir / "summary.parquet").unlink()

    result = generate_intraday_event_report(
        IntradayEventReportConfig(experiment_path=experiment_dir, top=2, charts=False)
    )

    assert result.top_run_count == 2
    assert (experiment_dir / "report.md").exists()


def test_intraday_event_report_missing_chart_columns_warns(tmp_path: Path) -> None:
    experiment_dir = make_report_experiment(tmp_path)
    summary = pl.read_parquet(experiment_dir / "summary.parquet").drop("window_minutes")
    summary.write_parquet(experiment_dir / "summary.parquet")
    summary.write_csv(experiment_dir / "summary.csv")

    result = generate_intraday_event_report(
        IntradayEventReportConfig(experiment_path=experiment_dir, top=2, charts=True)
    )

    assert any("window_minutes" in warning for warning in result.warnings)
    assert (experiment_dir / "report.md").exists()
