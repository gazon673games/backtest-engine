"""Builds searchable experiment registries and comparison reports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from backtest_engine.core.exceptions import ReportingError
from backtest_engine.research.experiments.compare import (
    best_by_group,
    ranked_top_runs,
    render_compare_report,
    sort_frame,
)
from backtest_engine.research.experiments.config import (
    ExperimentCompareConfig,
    ExperimentCompareResult,
    ExperimentRegistryConfig,
    ExperimentRegistryResult,
    OutputFormat,
)
from backtest_engine.research.experiments.io import (
    load_experiment_registry,
    output_file_hashes,
    output_formats,
    registry_frame,
    write_json,
    write_named_outputs,
)
from backtest_engine.research.experiments.rows import registry_rows
from backtest_engine.research.experiments.utils import status_count

__all__ = [
    "ExperimentCompareConfig",
    "ExperimentCompareResult",
    "ExperimentRegistryConfig",
    "ExperimentRegistryResult",
    "OutputFormat",
    "build_experiment_registry",
    "generate_experiment_compare_report",
    "load_experiment_registry",
]


def build_experiment_registry(config: ExperimentRegistryConfig) -> ExperimentRegistryResult:
    rows, experiment_ids, warnings = registry_rows(config.root)
    registry = registry_frame(rows)
    registry_path = write_registry_outputs(config, registry, experiment_ids, warnings)
    return ExperimentRegistryResult(
        root=config.root,
        output_path=config.output_path,
        registry_path=registry_path,
        run_count=registry.height,
        successful_run_count=status_count(registry, "success"),
        failed_run_count=status_count(registry, "failed"),
        experiment_count=len(experiment_ids),
        warnings=warnings,
    )


def generate_experiment_compare_report(
    config: ExperimentCompareConfig,
) -> ExperimentCompareResult:
    registry, registry_path, warnings = load_or_build_registry(config)
    successful = validated_successful_registry(registry, config.sort_by)
    sorted_successful = sort_frame(successful, config.sort_by, config.descending)
    top_runs = ranked_top_runs(sorted_successful, config.top)
    best_group, group_warnings = best_by_group(
        successful,
        config.group_by or ["continuous_symbol", "timeframe", "strategy_sha256"],
        config.sort_by,
        config.descending,
    )
    warnings.extend(group_warnings)
    report_path, top_runs_path, best_by_group_path = write_compare_outputs(
        config,
        registry,
        registry_path,
        top_runs,
        best_group,
        warnings,
    )
    return ExperimentCompareResult(
        output_path=config.output_path,
        report_path=report_path,
        registry_path=registry_path,
        top_runs_path=top_runs_path,
        best_by_group_path=best_by_group_path,
        run_count=registry.height,
        successful_run_count=successful.height,
        failed_run_count=status_count(registry, "failed"),
        top_run_count=top_runs.height,
        best_overall=top_runs.row(0, named=True) if top_runs.height > 0 else None,
        warnings=warnings,
    )


def write_registry_outputs(
    config: ExperimentRegistryConfig,
    registry: pl.DataFrame,
    experiment_ids: set[str],
    warnings: list[str],
) -> Path | None:
    if config.output_path is None:
        return None
    config.output_path.mkdir(parents=True, exist_ok=True)
    formats = output_formats(config.output_format)
    registry_path = write_named_outputs(
        config.output_path / "experiment_registry",
        registry,
        formats,
    )
    write_json(
        config.output_path / "registry_manifest.json",
        {
            "manifest_version": 1,
            "root": str(config.root),
            "generated_at": datetime.now(UTC).isoformat(),
            "run_count": registry.height,
            "successful_run_count": status_count(registry, "success"),
            "failed_run_count": status_count(registry, "failed"),
            "experiment_count": len(experiment_ids),
            "outputs": output_file_hashes(config.output_path, "experiment_registry"),
            "warnings": warnings,
        },
    )
    return registry_path


def load_or_build_registry(
    config: ExperimentCompareConfig,
) -> tuple[pl.DataFrame, Path | None, list[str]]:
    if config.registry_path is not None:
        return load_experiment_registry(config.registry_path), config.registry_path, []
    assert config.root is not None
    rows, _experiment_ids, warnings = registry_rows(config.root)
    return registry_frame(rows), None, warnings


def validated_successful_registry(registry: pl.DataFrame, sort_by: str) -> pl.DataFrame:
    if registry.height == 0:
        raise ReportingError("Experiment registry is empty")
    if sort_by not in registry.columns:
        msg = f"sort_by column does not exist in experiment registry: {sort_by}"
        raise ReportingError(msg)
    successful = registry.filter(pl.col("status") == "success")
    if successful.height == 0:
        raise ReportingError("Experiment registry has no successful runs to compare")
    return successful


def write_compare_outputs(
    config: ExperimentCompareConfig,
    registry: pl.DataFrame,
    registry_path: Path | None,
    top_runs: pl.DataFrame,
    best_group: pl.DataFrame,
    warnings: list[str],
) -> tuple[Path | None, Path | None, Path | None]:
    formats = output_formats(config.output_format)
    config.output_path.mkdir(parents=True, exist_ok=True)
    report_path = None
    if "md" in formats:
        report_path = config.output_path / "compare_report.md"
        report_path.write_text(
            render_compare_report(
                config=config,
                registry=registry,
                top_runs=top_runs,
                best_by_group=best_group,
                warnings=warnings,
            ),
            encoding="utf-8",
        )
    top_runs_path = write_named_outputs(
        config.output_path / "compare_top_runs",
        top_runs,
        formats,
    )
    best_by_group_path = write_named_outputs(
        config.output_path / "compare_best_by_group",
        best_group,
        formats,
    )
    write_compare_manifest(config, registry, registry_path, warnings)
    return report_path, top_runs_path, best_by_group_path


def write_compare_manifest(
    config: ExperimentCompareConfig,
    registry: pl.DataFrame,
    registry_path: Path | None,
    warnings: list[str],
) -> None:
    write_json(
        config.output_path / "compare_manifest.json",
        {
            "manifest_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "registry_path": str(registry_path) if registry_path is not None else None,
            "sort_by": config.sort_by,
            "descending": config.descending,
            "top": config.top,
            "group_by": config.group_by or ["continuous_symbol", "timeframe", "strategy_sha256"],
            "run_count": registry.height,
            "successful_run_count": status_count(registry, "success"),
            "failed_run_count": status_count(registry, "failed"),
            "outputs": output_file_hashes(config.output_path, "compare"),
            "warnings": warnings,
        },
    )
