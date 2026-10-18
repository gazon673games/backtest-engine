"""Runs walk-forward, robustness, and acceptance stages as one research pipeline."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backtest_engine.core.exceptions import BacktestError
from backtest_engine.intraday.event.batch.case_config import resolved_dataset_sources
from backtest_engine.intraday.event.batch.parsing import (
    expand_intraday_event_grid,
    split_intraday_event_params,
)
from backtest_engine.intraday.event.batch.runner import (
    IntradayEventBatchResult,
    load_intraday_event_batch_config,
    run_intraday_event_batch,
)
from backtest_engine.intraday.event.reporting.report import (
    IntradayEventReportConfig,
    IntradayEventReportResult,
    generate_intraday_event_report,
)
from backtest_engine.research.strategy_audit import StrategyAuditConfig, audit_strategy
from backtest_engine.research.validation.gate import (
    ResearchValidationConfig,
    ResearchValidationResult,
    generate_research_validation_report,
)


class IntradayEventResearchPipelineConfig(BaseModel):
    config_path: Path
    catalog_enforcement: str = "strict"
    report_sort_by: str = "sharpe"
    report_top: int = Field(default=20, gt=0)
    report_charts: bool = True
    split_date: date | None = None
    strategy_audit_policy: str = "strict"
    strategy_audit_sample_count: int = Field(default=32, ge=1)


class IntradayEventResearchPipelineResult(BaseModel):
    experiment_path: Path
    batch: IntradayEventBatchResult
    report: IntradayEventReportResult
    research_validation: ResearchValidationResult
    manifest_path: Path
    strategy_audit_paths: list[Path] = Field(default_factory=list)


def run_intraday_event_research_pipeline(
    config: IntradayEventResearchPipelineConfig,
) -> IntradayEventResearchPipelineResult:
    batch_config = load_intraday_event_batch_config(config.config_path)
    batch_config = batch_config.model_copy(
        update={"catalog_enforcement": config.catalog_enforcement}
    )
    audit_paths = run_pipeline_strategy_audits(batch_config, config)
    batch_result = run_intraday_event_batch(batch_config)
    if batch_result.successful_run_count == 0:
        raise BacktestError(
            "Intraday event pipeline produced zero successful runs; "
            f"failed_run_count={batch_result.failed_run_count}; "
            f"experiment_path={batch_result.output_dir}"
        )
    report_result = generate_intraday_event_report(
        IntradayEventReportConfig(
            experiment_path=batch_result.output_dir,
            sort_by=config.report_sort_by,
            top=config.report_top,
            charts=config.report_charts,
        )
    )
    research_result = generate_research_validation_report(
        ResearchValidationConfig(
            experiment_path=batch_result.output_dir,
            split_date=config.split_date,
        )
    )
    manifest_path = batch_result.output_dir / "research_pipeline_manifest.json"
    manifest = {
        "manifest_version": 1,
        "config_path": str(config.config_path),
        "catalog_enforcement": config.catalog_enforcement,
        "experiment_path": str(batch_result.output_dir),
        "successful_run_count": batch_result.successful_run_count,
        "failed_run_count": batch_result.failed_run_count,
        "report_path": str(report_result.report_path),
        "research_validation_report_path": str(research_result.report_path),
        "walk_forward_path": str(research_result.walk_forward_path),
        "rolling_walk_forward_path": str(research_result.rolling_walk_forward_path),
        "locked_holdout_path": str(research_result.locked_holdout_path),
        "research_validation_manifest_path": str(research_result.manifest_path),
        "strategy_audit_paths": [str(path) for path in audit_paths],
        "robustness_path": str(research_result.robustness_path),
        "warnings": [*report_result.warnings, *research_result.warnings],
    }
    _write_json(manifest_path, manifest)
    return IntradayEventResearchPipelineResult(
        experiment_path=batch_result.output_dir,
        batch=batch_result,
        report=report_result,
        research_validation=research_result,
        manifest_path=manifest_path,
        strategy_audit_paths=audit_paths,
    )


def run_pipeline_strategy_audits(
    batch_config: Any,
    pipeline_config: IntradayEventResearchPipelineConfig,
) -> list[Path]:
    policy = pipeline_config.strategy_audit_policy.strip().lower()
    if policy not in {"off", "audit", "strict"}:
        raise BacktestError("strategy_audit_policy must be off, audit, or strict")
    if policy == "off":
        return []
    plugin_parameter_sets = unique_plugin_parameter_sets(
        expand_intraday_event_grid(batch_config.grid)
    )
    output_dir = batch_config.output_root / batch_config.experiment_id / "strategy_audits"
    dataset_sources = resolved_dataset_sources(batch_config)
    paths = []
    for case_index, plugin_params in enumerate(plugin_parameter_sets):
        for dataset_index, (dataset_path, _) in enumerate(dataset_sources):
            result = audit_strategy(
                StrategyAuditConfig(
                    strategy_file=batch_config.strategy_file,
                    dataset_path=dataset_path,
                    mode="intraday_event",
                    params=plugin_params,
                    output_path=output_dir
                    / f"case-{case_index:03d}-dataset-{dataset_index:03d}.json",
                    sample_count=pipeline_config.strategy_audit_sample_count,
                    gate=policy == "strict",
                )
            )
            paths.append(result.output_path)
    return paths


def unique_plugin_parameter_sets(parameter_sets: list[dict[str, Any]]) -> list[dict[str, str]]:
    result = []
    seen = set()
    for parameter_set in parameter_sets:
        _, raw_params = split_intraday_event_params(parameter_set)
        params = {key: str(value) for key, value in raw_params.items()}
        key = json.dumps(params, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(params)
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
