from __future__ import annotations

import polars as pl

from backtest_engine.reporting.tables import markdown_table as markdown_table
from backtest_engine.research.validation.config import (
    DEFAULT_GROUP_COLUMNS,
    ResearchValidationConfig,
)
from backtest_engine.research.validation.stats import format_float


def render_report(
    *,
    config: ResearchValidationConfig,
    walk_forward: pl.DataFrame,
    robustness: pl.DataFrame,
    guardrails: pl.DataFrame,
    bootstrap: pl.DataFrame,
    block_bootstrap: pl.DataFrame,
    cost_stress: pl.DataFrame,
    regime: pl.DataFrame,
    rolling_walk_forward: pl.DataFrame,
    locked_holdout: pl.DataFrame,
    selection_bias: pl.DataFrame,
    deflated_sharpe: pl.DataFrame,
    warnings: list[str],
) -> str:
    lines = header(config, walk_forward, robustness, warnings)
    lines.extend(best_oos_section(walk_forward))
    lines.extend(table_section("Rolling Walk-Forward", rolling_walk_forward, "fold"))
    lines.extend(table_section("Locked Holdout", locked_holdout, "holdout_total_return"))
    lines.extend(selection_bias_section(selection_bias))
    lines.extend(
        table_section(
            "Deflated Sharpe",
            deflated_sharpe,
            "deflated_sharpe_probability",
        )
    )
    lines.extend(robustness_section(robustness))
    lines.extend(guardrails_section(guardrails))
    lines.extend(table_section("Bootstrap / Monte Carlo", bootstrap, "bootstrap_p05_total_return"))
    lines.extend(
        table_section("Block Bootstrap", block_bootstrap, "block_bootstrap_p05_total_return")
    )
    lines.extend(cost_stress_section(cost_stress))
    lines.extend(regime_section(regime))
    lines.extend(warning_section(warnings))
    return "\n".join(lines)


def header(
    config: ResearchValidationConfig,
    walk_forward: pl.DataFrame,
    robustness: pl.DataFrame,
    warnings: list[str],
) -> list[str]:
    return [
        "# Intraday Event Research Validation",
        "",
        f"- experiment_path: `{config.experiment_path}`",
        f"- run_count: {walk_forward.height}",
        f"- split_date: {config.split_date or 'auto 2/3 by trading dates'}",
        f"- robustness_group_count: {robustness.height}",
        f"- warning_count: {len(warnings)}",
        "",
    ]


def best_oos_section(walk_forward: pl.DataFrame) -> list[str]:
    if walk_forward.height == 0 or "oos_total_return" not in walk_forward.columns:
        return []
    best = walk_forward.sort("oos_total_return", descending=True, nulls_last=True).row(
        0,
        named=True,
    )
    return [
        "## Best OOS Run",
        "",
        f"- run_id: `{best.get('run_id')}`",
        f"- oos_total_return: {format_float(best.get('oos_total_return'))}",
        f"- is_total_return: {format_float(best.get('is_total_return'))}",
        f"- max_drawdown_oos: {format_float(best.get('oos_max_drawdown'))}",
        "",
    ]


def robustness_section(robustness: pl.DataFrame) -> list[str]:
    if robustness.height == 0:
        return []
    lines = ["## Robustness", ""]
    for row in robustness.head(10).iter_rows(named=True):
        label = ", ".join(f"{key}={row[key]}" for key in DEFAULT_GROUP_COLUMNS if key in row)
        lines.append(
            "- "
            f"{label}: median_oos={format_float(row.get('median_oos_total_return'))}, "
            f"min_oos={format_float(row.get('min_oos_total_return'))}, "
            f"runs={row.get('run_count')}"
        )
    lines.append("")
    return lines


def guardrails_section(guardrails: pl.DataFrame) -> list[str]:
    if guardrails.height == 0:
        return []
    row = guardrails.row(0, named=True)
    return [
        "## Optimization Guardrails",
        "",
        f"- multiple_testing_warning: `{row.get('multiple_testing_warning')}`",
        f"- rank_instability_warning: `{row.get('rank_instability_warning')}`",
        f"- oos_degradation_warning: `{row.get('oos_degradation_warning')}`",
        f"- best_is_run_id: `{row.get('best_is_run_id')}`",
        f"- best_oos_run_id: `{row.get('best_oos_run_id')}`",
        "- best_is_oos_to_is_return_ratio: "
        f"{format_float(row.get('best_is_oos_to_is_return_ratio'))}",
        f"- negative_oos_run_share: {format_float(row.get('negative_oos_run_share'))}",
        "",
    ]


def table_section(title: str, frame: pl.DataFrame, sort_by: str) -> list[str]:
    if frame.height == 0:
        return []
    top = frame.sort(sort_by, descending=True, nulls_last=True).head(10)
    return [f"## {title}", "", markdown_table(top), ""]


def cost_stress_section(cost_stress: pl.DataFrame) -> list[str]:
    if cost_stress.height == 0:
        return []
    top_cost = cost_stress.sort(
        ["extra_cost_bps", "stressed_total_return"],
        descending=[False, True],
        nulls_last=True,
    ).head(20)
    return ["## Cost Stress", "", markdown_table(top_cost), ""]


def regime_section(regime: pl.DataFrame) -> list[str]:
    if regime.height == 0:
        return []
    monthly = regime.filter(pl.col("regime_type") == "month").sort(
        "total_return",
        descending=True,
        nulls_last=True,
    )
    return ["## Regime Breakdown", "", markdown_table(monthly.head(20)), ""]


def warning_section(warnings: list[str]) -> list[str]:
    if not warnings:
        return []
    return ["## Warnings", "", *(f"- {warning}" for warning in warnings), ""]


def selection_bias_section(frame: pl.DataFrame) -> list[str]:
    if frame.height == 0:
        return []
    row = frame.row(0, named=True)
    return [
        "## Selection Bias Controls",
        "",
        f"- trial_count: {row.get('trial_count')}",
        "- best_deflated_sharpe_probability: "
        f"{format_float(row.get('best_deflated_sharpe_probability'))}",
        "- probability_of_backtest_overfitting: "
        f"{format_float(row.get('probability_of_backtest_overfitting'))}",
        f"- reality_check_p_value: {format_float(row.get('reality_check_p_value'))}",
        "",
    ]
