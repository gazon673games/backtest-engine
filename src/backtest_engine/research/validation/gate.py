"""Turns statistical and data checks into formal research acceptance decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import ReportingError
from backtest_engine.research.selection_bias import analyze_selection_bias
from backtest_engine.research.validation.analytics import (
    block_bootstrap_metrics,
    bootstrap_metrics,
    build_guardrails,
    build_robustness,
    cost_stress_metrics,
    regime_metrics,
)
from backtest_engine.research.validation.config import (
    ResearchValidationConfig,
    ResearchValidationResult,
)
from backtest_engine.research.validation.io import read_table, write_table
from backtest_engine.research.validation.markdown import render_report
from backtest_engine.research.validation.rolling import rolling_validation
from backtest_engine.research.validation.stats import (
    default_split_date,
    ratio,
    segment_metrics,
    segment_total_return,
)

__all__ = [
    "ResearchValidationConfig",
    "ResearchValidationResult",
    "generate_research_validation_report",
]


def generate_research_validation_report(
    config: ResearchValidationConfig,
) -> ResearchValidationResult:
    output_path = config.output_path or config.experiment_path / "research_validation"
    output_path.mkdir(parents=True, exist_ok=True)
    summary = read_table(config.experiment_path / "summary")
    walk_forward, derived, warnings = build_research_frames(config, summary)
    paths = write_research_outputs(output_path, walk_forward, derived)
    gate = evaluate_research_gate(
        config,
        derived["guardrails"],
        warnings,
        selection_bias=derived["selection_bias"],
    )
    gate_path = output_path / "research_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if config.gate and gate["acceptance_status"] == "rejected":
        raise ReportingError("Research gate rejected result: " + "; ".join(gate["reasons"]))
    report_path = output_path / "report.md"
    report_path.write_text(
        render_report(
            config=config,
            walk_forward=walk_forward,
            robustness=derived["robustness"],
            guardrails=derived["guardrails"],
            bootstrap=derived["bootstrap"],
            block_bootstrap=derived["block_bootstrap"],
            cost_stress=derived["cost_stress"],
            regime=derived["regime"],
            rolling_walk_forward=derived["rolling_walk_forward"],
            locked_holdout=derived["locked_holdout"],
            selection_bias=derived["selection_bias"],
            deflated_sharpe=derived["deflated_sharpe"],
            warnings=warnings,
        ),
        encoding="utf-8",
    )
    manifest_path = output_path / "research_validation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "config": config.model_dump(mode="json"),
                "outputs": {
                    path.name: file_sha256(path)
                    for path in [*paths.values(), gate_path, report_path]
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ResearchValidationResult(
        output_path=output_path,
        walk_forward_path=paths["walk_forward"],
        rolling_walk_forward_path=paths["rolling_walk_forward"],
        locked_holdout_path=paths["locked_holdout"],
        robustness_path=paths["robustness"],
        guardrails_path=paths["guardrails"],
        gate_path=gate_path,
        bootstrap_path=paths["bootstrap"],
        block_bootstrap_path=paths["block_bootstrap"],
        cost_stress_path=paths["cost_stress"],
        regime_path=paths["regime"],
        selection_bias_path=paths["selection_bias"],
        deflated_sharpe_path=paths["deflated_sharpe"],
        report_path=report_path,
        manifest_path=manifest_path,
        run_count=walk_forward.height,
        warning_count=len(warnings),
        acceptance_status=str(gate["acceptance_status"]),
        warnings=warnings,
    )


def build_research_frames(
    config: ResearchValidationConfig,
    summary: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, pl.DataFrame], list[str]]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    block_bootstrap_rows: list[dict[str, Any]] = []
    cost_stress_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    run_events = []

    for run_index, row in enumerate(summary.iter_rows(named=True)):
        events = read_run_events(row, warnings)
        if events is None:
            continue
        run_events.append((row, events))
        split_date = config.split_date or default_split_date(events)
        rows.append(walk_forward_row(row, events, split_date))
        append_optional(bootstrap_rows, bootstrap_metrics(row, events, config, run_index))
        append_optional(
            block_bootstrap_rows,
            block_bootstrap_metrics(row, events, config, run_index),
        )
        cost_stress_rows.extend(cost_stress_metrics(row, events, config))
        regime_rows.extend(regime_metrics(row, events))

    walk_forward = frame_from_rows(rows)
    robustness = build_robustness(walk_forward, config.group_columns)
    guardrails, guardrail_warnings = build_guardrails(walk_forward, config)
    warnings.extend(guardrail_warnings)
    rolling_walk_forward, locked_holdout, rolling_warnings = rolling_validation(run_events, config)
    warnings.extend(rolling_warnings)
    selection_bias = analyze_selection_bias(
        run_events,
        bootstrap_iterations=config.selection_bias_bootstrap_iterations,
        seed=config.selection_bias_seed,
        pbo_partitions=config.pbo_partitions,
    )
    return (
        walk_forward,
        {
            "robustness": robustness,
            "guardrails": guardrails,
            "bootstrap": frame_from_rows(bootstrap_rows),
            "block_bootstrap": frame_from_rows(block_bootstrap_rows),
            "cost_stress": frame_from_rows(cost_stress_rows),
            "regime": frame_from_rows(regime_rows),
            "rolling_walk_forward": rolling_walk_forward,
            "locked_holdout": locked_holdout,
            "selection_bias": selection_bias.summary,
            "deflated_sharpe": selection_bias.deflated_sharpe,
        },
        warnings,
    )


def read_run_events(row: dict[str, Any], warnings: list[str]) -> pl.DataFrame | None:
    run_path = Path(str(row.get("run_path") or ""))
    events_path = run_path / "events.parquet"
    if not events_path.exists():
        warnings.append(f"missing events.parquet for run_id={row.get('run_id')}")
        return None
    events = pl.read_parquet(events_path)
    if events.height == 0:
        warnings.append(f"empty events for run_id={row.get('run_id')}")
        return None
    return events


def walk_forward_row(
    row: dict[str, Any],
    events: pl.DataFrame,
    split_date: Any,
) -> dict[str, Any]:
    in_sample = events.filter(pl.col("ts").dt.date() < split_date)
    out_of_sample = events.filter(pl.col("ts").dt.date() >= split_date)
    return {
        **row,
        "split_date": split_date.isoformat(),
        **segment_metrics("is", in_sample),
        **segment_metrics("oos", out_of_sample),
        "oos_to_is_return_ratio": ratio(
            segment_total_return(out_of_sample),
            segment_total_return(in_sample),
        ),
    }


def write_research_outputs(
    output_path: Path,
    walk_forward: pl.DataFrame,
    derived: dict[str, pl.DataFrame],
) -> dict[str, Path]:
    frames = {"walk_forward": walk_forward, **derived}
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output_path / f"{name}.csv"
        paths[name] = path
        write_table(path.with_suffix(""), frame)
    return paths


def frame_from_rows(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, orient="row") if rows else pl.DataFrame()


def append_optional(rows: list[dict[str, Any]], row: dict[str, Any] | None) -> None:
    if row is not None:
        rows.append(row)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_research_gate(
    config: ResearchValidationConfig,
    guardrails: pl.DataFrame,
    warnings: list[str],
    selection_bias: pl.DataFrame | None = None,
) -> dict[str, Any]:
    reasons = []
    if len(warnings) > config.max_warning_count:
        reasons.append(f"warning_count {len(warnings)} > {config.max_warning_count}")
    if guardrails.height > 0:
        row = guardrails.row(0, named=True)
        if bool(row.get("multiple_testing_warning")):
            reasons.append("multiple_testing_warning")
        if bool(row.get("rank_instability_warning")):
            reasons.append("rank_instability_warning")
        if bool(row.get("oos_degradation_warning")):
            reasons.append("oos_degradation_warning")
        negative_share = row.get("negative_oos_run_share")
        if negative_share is not None and negative_share > config.max_negative_oos_run_share:
            reasons.append(
                f"negative_oos_run_share {negative_share} > {config.max_negative_oos_run_share}"
            )
    if selection_bias is not None and selection_bias.height:
        bias = selection_bias.row(0, named=True)
        if int(bias.get("trial_count") or 0) > 1:
            pbo = bias.get("probability_of_backtest_overfitting")
            if pbo is not None and pbo > config.max_probability_of_backtest_overfitting:
                reasons.append(
                    "probability_of_backtest_overfitting "
                    f"{pbo} > {config.max_probability_of_backtest_overfitting}"
                )
            reality_p = bias.get("reality_check_p_value")
            if reality_p is not None and reality_p > config.max_reality_check_p_value:
                reasons.append(
                    f"reality_check_p_value {reality_p} > {config.max_reality_check_p_value}"
                )
            deflated = bias.get("best_deflated_sharpe_probability")
            if deflated is not None and deflated < config.min_deflated_sharpe_probability:
                reasons.append(
                    "best_deflated_sharpe_probability "
                    f"{deflated} < {config.min_deflated_sharpe_probability}"
                )
    return {
        "gate_enabled": config.gate,
        "acceptance_status": "rejected" if reasons else "accepted",
        "reasons": reasons,
        "warning_count": len(warnings),
        "max_warning_count": config.max_warning_count,
        "max_negative_oos_run_share": config.max_negative_oos_run_share,
        "max_probability_of_backtest_overfitting": (config.max_probability_of_backtest_overfitting),
        "max_reality_check_p_value": config.max_reality_check_p_value,
        "min_deflated_sharpe_probability": config.min_deflated_sharpe_probability,
    }
