from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from backtest_engine.cli import app
from backtest_engine.research.pack import ResearchPackConfig, generate_research_pack

runner = CliRunner()


def test_research_pack_writes_run_markdown_and_html(tmp_path: Path) -> None:
    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "metrics.json").write_text(
        json.dumps({"total_return": 0.12, "sharpe": 1.5}),
        encoding="utf-8",
    )
    (run_path / "experiment_manifest.json").write_text(
        json.dumps({"engine": "fixture"}),
        encoding="utf-8",
    )

    result = generate_research_pack(
        ResearchPackConfig(path=run_path, output_path=tmp_path / "pack", generate_tearsheet=False)
    )

    assert result.pack_type == "run"
    assert result.markdown_path.exists()
    assert result.html_path.exists()
    assert "Run Metrics" in result.markdown_path.read_text(encoding="utf-8")


def test_research_pack_writes_experiment_top_runs(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment"
    experiment_path.mkdir()
    pl.DataFrame(
        {
            "run_id": ["a", "b"],
            "total_return": [0.1, 0.2],
            "run_path": ["runs/a", "runs/b"],
        }
    ).write_parquet(experiment_path / "summary.parquet")
    pl.DataFrame(
        {
            "run_id": ["bad"],
            "error_message": ["fixture failure"],
        }
    ).write_parquet(experiment_path / "failed_runs.parquet")

    result = generate_research_pack(ResearchPackConfig(path=experiment_path, top=1))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.pack_type == "experiment"
    assert manifest["top_runs"][0]["run_id"] == "b"
    assert "Failed Runs" in result.markdown_path.read_text(encoding="utf-8")


def test_cli_research_pack_smoke(tmp_path: Path) -> None:
    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "metrics.json").write_text(json.dumps({"total_return": 0.01}), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "research-pack",
            "--path",
            str(run_path),
            "--output-path",
            str(tmp_path / "pack"),
            "--no-tearsheet",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "pack_type=run" in result.stdout
    assert (tmp_path / "pack" / "research_pack.md").exists()
