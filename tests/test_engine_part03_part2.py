# ruff: noqa: F405
from test_engine_support import *  # noqa: F401,F403


def test_report_best_run_per_dataset(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path)

    result = generate_experiment_report(ReportConfig(experiment_path=experiment_path))
    best = pl.read_json(experiment_path / "best_by_dataset.json")

    assert {row["continuous_symbol"]: row["run_id"] for row in result.best_by_dataset} == {
        "CNYRUB_CONT": "cny_fast",
        "USDRUB_CONT": "usd_fast",
    }
    assert best.height == 2


def test_report_failed_runs_included(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path, include_failed=True)

    result = generate_experiment_report(ReportConfig(experiment_path=experiment_path))
    report_text = (experiment_path / "report.md").read_text(encoding="utf-8")

    assert result.failed_run_count == 1
    assert "blocked_by_quality_gate" in report_text
    assert "Dataset is not approved" in report_text


def test_report_writes_markdown_and_top_files(tmp_path: Path) -> None:
    experiment_path = write_experiment_outputs(tmp_path)

    generate_experiment_report(ReportConfig(experiment_path=experiment_path))

    assert (experiment_path / "report.md").exists()
    assert (experiment_path / "top_runs.csv").exists()
    assert (experiment_path / "top_runs.parquet").exists()
    assert (experiment_path / "top_runs.json").exists()
    assert (experiment_path / "best_by_dataset.csv").exists()
    assert (experiment_path / "best_by_dataset.parquet").exists()
    assert (experiment_path / "best_by_dataset.json").exists()
