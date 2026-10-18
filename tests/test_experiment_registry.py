from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from backtest_engine.cli import app
from backtest_engine.research.experiments.registry import (
    ExperimentCompareConfig,
    ExperimentRegistryConfig,
    build_experiment_registry,
    generate_experiment_compare_report,
)

runner = CliRunner()


def test_experiment_registry_indexes_run_manifests_and_failed_rows(tmp_path: Path) -> None:
    root = write_registry_fixture(tmp_path)
    output_path = tmp_path / "registry"

    result = build_experiment_registry(ExperimentRegistryConfig(root=root, output_path=output_path))
    registry = pl.read_parquet(output_path / "experiment_registry.parquet").sort(
        ["status", "run_id"], nulls_last=True
    )

    assert result.run_count == 3
    assert result.successful_run_count == 2
    assert result.failed_run_count == 1
    assert result.experiment_count == 1
    assert registry.filter(pl.col("status") == "success").height == 2
    assert (
        registry.filter(pl.col("status") == "failed").row(0, named=True)["error_type"]
        == "DatasetValidationError"
    )
    assert (
        registry.filter(pl.col("run_id") == "run_b").row(0, named=True)["parent_experiment_id"]
        == "registry_fixture"
    )
    assert (output_path / "registry_manifest.json").exists()


def test_experiment_compare_report_ranks_and_groups_registry(tmp_path: Path) -> None:
    root = write_registry_fixture(tmp_path)
    registry_output = tmp_path / "registry"
    compare_output = tmp_path / "compare"
    build_experiment_registry(ExperimentRegistryConfig(root=root, output_path=registry_output))

    result = generate_experiment_compare_report(
        ExperimentCompareConfig(
            registry_path=registry_output / "experiment_registry.parquet",
            output_path=compare_output,
            sort_by="sharpe",
            top=1,
            group_by=["continuous_symbol"],
        )
    )
    top_runs = pl.read_parquet(compare_output / "compare_top_runs.parquet")
    best_by_group = pl.read_parquet(compare_output / "compare_best_by_group.parquet")

    assert result.top_run_count == 1
    assert result.best_overall is not None
    assert result.best_overall["run_id"] == "run_b"
    assert top_runs.get_column("run_id").to_list() == ["run_b"]
    assert best_by_group.get_column("group_key").to_list() == [
        "continuous_symbol=CNYRUB_NEAREST_5M"
    ]
    assert (compare_output / "compare_report.md").exists()
    assert (compare_output / "compare_manifest.json").exists()


def test_experiment_registry_cli_smoke(tmp_path: Path) -> None:
    root = write_registry_fixture(tmp_path)
    registry_output = tmp_path / "registry_cli"

    result = runner.invoke(
        app,
        [
            "experiment-index",
            "--root",
            str(root),
            "--output-path",
            str(registry_output),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "run_count=3" in result.stdout
    assert (registry_output / "experiment_registry.parquet").exists()


def test_experiment_compare_cli_smoke(tmp_path: Path) -> None:
    root = write_registry_fixture(tmp_path)
    registry_output = tmp_path / "registry"
    compare_output = tmp_path / "compare_cli"
    build_experiment_registry(ExperimentRegistryConfig(root=root, output_path=registry_output))

    result = runner.invoke(
        app,
        [
            "experiment-compare",
            "--registry-path",
            str(registry_output / "experiment_registry.parquet"),
            "--output-path",
            str(compare_output),
            "--sort-by",
            "sharpe",
            "--top",
            "1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "top_run_count=1" in result.stdout
    assert (compare_output / "compare_report.md").exists()


def write_registry_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "backtests"
    experiment = root / "registry_fixture"
    run_a = experiment / "runs" / "run_a"
    run_b = experiment / "runs" / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    write_run_manifest(run_a, run_id="run_a", sharpe=0.7, total_return=0.03)
    write_run_manifest(run_b, run_id="run_b", sharpe=1.4, total_return=0.05)
    write_batch_manifest(experiment, [run_a, run_b])
    return root


def write_run_manifest(
    run_path: Path,
    *,
    run_id: str,
    sharpe: float,
    total_return: float,
) -> None:
    payload = {
        "manifest_version": 1,
        "run_id": run_id,
        "dataset": {
            "path": "data/cnyrub.parquet",
            "sha256": "dataset-sha",
            "validation": {
                "continuous_symbol": "CNYRUB_NEAREST_5M",
                "timeframe": "5m",
                "dataset_type": "nearest_contract",
                "adjustment_mode": "unadjusted",
            },
        },
        "catalog_approval": {"status": "approved"},
        "quality_approval": {"status": "approved"},
        "strategy": {
            "path": "strategies/fixture.py",
            "sha256": "strategy-sha",
            "params": {"mode": run_id},
        },
        "portfolio": {"initial_capital": 1000.0, "sizing_mode": "fixed_contracts"},
        "execution": {
            "schedule": "times",
            "decision_times": ["13:30"],
            "from_date": "2024-01-02",
            "timing": "signal_close_to_next_bar_close",
            "missing_decision_policy": "carry",
            "commission_bps": 1.0,
            "slippage_bps": 0.5,
            "slippage_ticks": 0.0,
            "bid_ask_spread_bps": 2.0,
            "bid_ask_spread_ticks": 0.0,
            "volume_participation_limit": None,
            "partial_fill_policy": "allow",
            "intrabar_ambiguity_policy": "conservative",
        },
        "metrics": {
            "event_count": 10,
            "trade_count": 2,
            "order_count": 2,
            "partial_fill_count": 0,
            "volume_rejected_count": 0,
            "total_return": total_return,
            "annualized_return": total_return * 10,
            "annualized_volatility": 0.2,
            "sharpe": sharpe,
            "max_drawdown": -0.01,
            "net_pnl": total_return * 1000.0,
            "final_equity_cash": 1000.0 + total_return * 1000.0,
            "total_cost_cash": 1.5,
            "turnover_notional": 10000.0,
            "max_position_notional": 5000.0,
            "exposure": 1.0,
            "long_exposure": 1.0,
            "short_exposure": 0.0,
        },
    }
    (run_path / "experiment_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_batch_manifest(experiment: Path, run_paths: list[Path]) -> None:
    payload = {
        "manifest_version": 1,
        "experiment_id": "registry_fixture",
        "output_dir": str(experiment),
        "successful_run_count": len(run_paths),
        "failed_run_count": 1,
        "run_manifests": [
            {
                "run_id": run_path.name,
                "run_path": str(run_path),
                "manifest_path": str(run_path / "experiment_manifest.json"),
                "manifest_sha256": "run-manifest-sha",
            }
            for run_path in run_paths
        ],
        "failed_runs": [
            {
                "dataset_path": "data/broken.parquet",
                "strategy_file": "strategies/fixture.py",
                "params": '{"mode":"broken"}',
                "status": "failed",
                "error_type": "DatasetValidationError",
                "error_message": "fixture failure",
            }
        ],
    }
    (experiment / "experiment_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
