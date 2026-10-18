# ruff: noqa
from test_engine_support import *  # noqa: F401,F403


def test_report_creates_equity_and_drawdown_charts(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path)

    result = generate_experiment_report(ReportConfig(experiment_path=experiment_path))

    assert "charts/best_overall_equity.png" in result.chart_paths
    assert "charts/best_overall_drawdown.png" in result.chart_paths
    assert (experiment_path / "charts" / "best_overall_equity.png").exists()
    assert (experiment_path / "charts" / "best_overall_drawdown.png").exists()


def test_report_creates_heatmap_chart_from_summary(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path)

    result = generate_experiment_report(
        ReportConfig(experiment_path=experiment_path, chart_metric="sharpe")
    )

    assert "charts/CNYRUB_CONT_sharpe_heatmap.png" in result.chart_paths
    assert "charts/USDRUB_CONT_sharpe_heatmap.png" in result.chart_paths
    assert (experiment_path / "charts" / "CNYRUB_CONT_sharpe_heatmap.png").exists()
    assert (experiment_path / "charts" / "USDRUB_CONT_sharpe_heatmap.png").exists()


def test_report_command_creates_charts_directory_and_manifest(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path)

    result = runner.invoke(
        app,
        [
            "report",
            "--experiment-path",
            str(experiment_path),
            "--sort-by",
            "sharpe",
            "--top",
            "2",
            "--charts",
        ],
    )

    assert result.exit_code == 0
    assert (experiment_path / "charts").is_dir()
    assert (experiment_path / "charts" / "chart_manifest.json").exists()


def test_report_markdown_includes_chart_links(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path)

    generate_experiment_report(ReportConfig(experiment_path=experiment_path))
    report_text = (experiment_path / "report.md").read_text(encoding="utf-8")

    assert "## Charts" in report_text
    assert "charts/best_overall_equity.png" in report_text
    assert "charts/CNYRUB_CONT_sharpe_heatmap.png" in report_text


def test_report_no_charts_skips_chart_creation(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path)

    result = runner.invoke(
        app,
        [
            "report",
            "--experiment-path",
            str(experiment_path),
            "--no-charts",
        ],
    )

    assert result.exit_code == 0
    assert not (experiment_path / "charts").exists()


def test_report_missing_heatmap_columns_warns_instead_of_crashing(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path)
    summary = pl.read_parquet(experiment_path / "summary.parquet").drop("fast_window")
    summary.write_parquet(experiment_path / "summary.parquet")
    summary.write_csv(experiment_path / "summary.csv")
    (experiment_path / "summary.json").write_text(
        json.dumps(summary.to_dicts()),
        encoding="utf-8",
    )

    result = generate_experiment_report(ReportConfig(experiment_path=experiment_path))
    report_text = (experiment_path / "report.md").read_text(encoding="utf-8")

    assert result.top_run_count == 4
    assert any("heatmap" in warning for warning in result.warnings)
    assert "Chart Warnings" in report_text
    assert "missing columns: fast_window" in report_text


def test_cli_report_smoke_test(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path)

    result = runner.invoke(
        app,
        [
            "report",
            "--experiment-path",
            str(experiment_path),
            "--sort-by",
            "sharpe",
            "--top",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "top_run_count=2" in result.stdout
    assert (experiment_path / "report.md").exists()
