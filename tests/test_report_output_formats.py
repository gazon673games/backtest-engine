from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from backtest_engine.reporting.experiment import ReportConfig, generate_experiment_report


@pytest.mark.parametrize("output_format", ["csv", "parquet", "json"])
def test_report_paths_point_to_the_selected_output_format(
    tmp_path: Path, output_format: str
) -> None:
    pl.DataFrame({"continuous_symbol": ["TEST"], "run_id": ["run"], "sharpe": [1.0]}).write_parquet(
        tmp_path / "summary.parquet"
    )

    result = generate_experiment_report(
        ReportConfig(experiment_path=tmp_path, output_format=output_format, charts=False)
    )

    for path in (result.top_runs_path, result.best_by_dataset_path):
        assert path is not None
        assert path.suffix == f".{output_format}"
        assert path.is_file()
