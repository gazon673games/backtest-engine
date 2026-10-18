from __future__ import annotations

import random
from itertools import repeat
from math import prod
from typing import Any

import polars as pl

from backtest_engine.research.validation.config import (
    DEFAULT_GROUP_COLUMNS,
    ResearchValidationConfig,
)
from backtest_engine.research.validation.stats import (
    compound_values,
    equity_from_returns,
    float_values,
    format_float,
    max_column,
    max_drawdown,
    median_column,
    min_column,
    optional_float,
    percentile,
    stress_turnover,
)


def build_robustness(walk_forward: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
    if walk_forward.height == 0:
        return pl.DataFrame()
    available_groups = [
        column
        for column in (group_columns or list(DEFAULT_GROUP_COLUMNS))
        if column in walk_forward.columns
    ]
    if not available_groups:
        return pl.DataFrame()
    rows = []
    for key, frame in walk_forward.group_by(available_groups, maintain_order=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(available_groups, key_values, strict=True))
        row.update(robustness_row(frame))
        rows.append(row)
    return pl.DataFrame(rows, orient="row").sort("median_oos_total_return", descending=True)


def robustness_row(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "run_count": frame.height,
        "oos_run_count": int(frame.get_column("oos_total_return").drop_nulls().len()),
        "min_oos_total_return": min_column(frame, "oos_total_return"),
        "median_oos_total_return": median_column(frame, "oos_total_return"),
        "max_oos_total_return": max_column(frame, "oos_total_return"),
        "min_oos_to_is_return_ratio": min_column(frame, "oos_to_is_return_ratio"),
        "median_oos_to_is_return_ratio": median_column(frame, "oos_to_is_return_ratio"),
        "max_oos_unfilled_order_count": max_column(frame, "oos_unfilled_order_count"),
    }


def build_guardrails(
    walk_forward: pl.DataFrame,
    config: ResearchValidationConfig,
) -> tuple[pl.DataFrame, list[str]]:
    if walk_forward.height == 0:
        return pl.DataFrame(), []
    best_is = walk_forward.sort("is_total_return", descending=True, nulls_last=True).row(
        0,
        named=True,
    )
    best_oos = walk_forward.sort("oos_total_return", descending=True, nulls_last=True).row(
        0,
        named=True,
    )
    warnings = guardrail_warnings(walk_forward, best_is, best_oos, config)
    guardrails = pl.DataFrame(
        [guardrail_row(walk_forward, best_is, best_oos, config)],
        orient="row",
    )
    return guardrails, warnings


def guardrail_row(
    walk_forward: pl.DataFrame,
    best_is: dict[str, Any],
    best_oos: dict[str, Any],
    config: ResearchValidationConfig,
) -> dict[str, Any]:
    best_is_oos_return = optional_float(best_is.get("oos_total_return"))
    best_is_is_return = optional_float(best_is.get("is_total_return"))
    best_is_ratio = optional_float(best_is.get("oos_to_is_return_ratio"))
    return {
        "run_count": walk_forward.height,
        "max_parameter_runs_without_warning": config.max_parameter_runs_without_warning,
        "multiple_testing_warning": walk_forward.height > config.max_parameter_runs_without_warning,
        "best_is_run_id": best_is.get("run_id"),
        "best_oos_run_id": best_oos.get("run_id"),
        "rank_instability_warning": best_is.get("run_id") != best_oos.get("run_id"),
        "best_is_total_return": best_is_is_return,
        "best_is_oos_total_return": best_is_oos_return,
        "best_oos_total_return": optional_float(best_oos.get("oos_total_return")),
        "best_is_oos_to_is_return_ratio": best_is_ratio,
        "min_best_oos_to_is_return_ratio": config.min_best_oos_to_is_return_ratio,
        "oos_degradation_warning": oos_degradation_warning(best_is, config),
        "negative_oos_run_share": negative_oos_share(walk_forward),
    }


def guardrail_warnings(
    walk_forward: pl.DataFrame,
    best_is: dict[str, Any],
    best_oos: dict[str, Any],
    config: ResearchValidationConfig,
) -> list[str]:
    warnings = []
    if walk_forward.height > config.max_parameter_runs_without_warning:
        warnings.append(
            "parameter search has many runs: "
            f"{walk_forward.height} > {config.max_parameter_runs_without_warning}"
        )
    if oos_degradation_warning(best_is, config):
        warnings.append(
            "best in-sample run degrades out-of-sample: "
            f"run_id={best_is.get('run_id')}, "
            f"is_return={format_float(optional_float(best_is.get('is_total_return')))}, "
            f"oos_return={format_float(optional_float(best_is.get('oos_total_return')))}"
        )
    if best_is.get("run_id") != best_oos.get("run_id"):
        warnings.append(
            "best in-sample run differs from best out-of-sample run: "
            f"is={best_is.get('run_id')}, oos={best_oos.get('run_id')}"
        )
    return warnings


def bootstrap_metrics(
    summary_row: dict[str, Any],
    events: pl.DataFrame,
    config: ResearchValidationConfig,
    run_index: int,
) -> dict[str, Any] | None:
    returns = clean_returns(events, config.bootstrap_iterations)
    if not returns:
        return None
    rng = random.Random(config.bootstrap_seed + run_index)
    for index, value in enumerate(returns):
        returns[index] = 1.0 + value
    samples = []
    for _ in range(config.bootstrap_iterations):
        samples.append(prod(map(rng.choice, repeat(returns, len(returns))), start=1.0) - 1.0)
    samples.sort()
    return {
        "run_id": summary_row.get("run_id"),
        "bootstrap_iterations": config.bootstrap_iterations,
        "bootstrap_mean_total_return": sum(samples) / len(samples),
        "bootstrap_p05_total_return": percentile(samples, 0.05),
        "bootstrap_p50_total_return": percentile(samples, 0.50),
        "bootstrap_p95_total_return": percentile(samples, 0.95),
        "bootstrap_probability_loss": sum(1 for value in samples if value < 0.0) / len(samples),
    }


def block_bootstrap_metrics(
    summary_row: dict[str, Any],
    events: pl.DataFrame,
    config: ResearchValidationConfig,
    run_index: int,
) -> dict[str, Any] | None:
    returns = clean_returns(events, config.block_bootstrap_iterations)
    if not returns:
        return None
    block_length = min(config.block_bootstrap_length, len(returns))
    for index, value in enumerate(returns):
        returns[index] = 1.0 + value
    block_count = len(returns) - block_length + 1
    rng = random.Random(config.bootstrap_seed + 10_000 + run_index)
    samples = []
    for _ in range(config.block_bootstrap_iterations):
        equity = 1.0
        for offset in range(0, len(returns), block_length):
            start = rng.randrange(block_count)
            take = min(block_length, len(returns) - offset)
            equity = prod(returns[start : start + take], start=equity)
        samples.append(equity - 1.0)
    samples.sort()
    return {
        "run_id": summary_row.get("run_id"),
        "block_bootstrap_iterations": config.block_bootstrap_iterations,
        "block_length": block_length,
        "block_bootstrap_mean_total_return": sum(samples) / len(samples),
        "block_bootstrap_p05_total_return": percentile(samples, 0.05),
        "block_bootstrap_p50_total_return": percentile(samples, 0.50),
        "block_bootstrap_p95_total_return": percentile(samples, 0.95),
        "block_bootstrap_probability_loss": sum(1 for value in samples if value < 0.0)
        / len(samples),
    }


def cost_stress_metrics(
    summary_row: dict[str, Any],
    events: pl.DataFrame,
    config: ResearchValidationConfig,
) -> list[dict[str, Any]]:
    if events.height == 0 or "net_return" not in events.columns:
        return []
    base_returns = [float(value) for value in events.get_column("net_return").to_list()]
    base_total_return = compound_values(base_returns) or 0.0
    turnover = stress_turnover(events)
    rows = []
    for stress_bps in config.cost_stress_bps:
        stressed_returns = [
            base_return - turn * stress_bps / 10_000.0
            for base_return, turn in zip(base_returns, turnover, strict=True)
        ]
        stressed_total_return = compound_values(stressed_returns)
        rows.append(
            {
                "run_id": summary_row.get("run_id"),
                "extra_cost_bps": stress_bps,
                "stressed_total_return": stressed_total_return,
                "stressed_max_drawdown": max_drawdown(equity_from_returns(stressed_returns)),
                "stressed_loss": base_total_return - (stressed_total_return or 0.0),
            }
        )
    return rows


def regime_metrics(summary_row: dict[str, Any], events: pl.DataFrame) -> list[dict[str, Any]]:
    if events.height == 0 or "net_return" not in events.columns or "ts" not in events.columns:
        return []
    rows = []
    working = events.select("ts", "net_return").with_columns(
        pl.col("ts").dt.strftime("%Y").alias("year"),
        pl.col("ts").dt.strftime("%Y-%m").alias("month"),
        pl.col("ts").dt.weekday().alias("weekday"),
    )
    for regime_type in ("year", "month", "weekday"):
        rows.extend(regime_group_rows(summary_row, working, regime_type))
    return rows


def regime_group_rows(
    summary_row: dict[str, Any],
    frame: pl.DataFrame,
    regime_type: str,
) -> list[dict[str, Any]]:
    rows = []
    for regime_value, group in frame.group_by(regime_type, maintain_order=True):
        value = regime_value[0] if isinstance(regime_value, tuple) else regime_value
        returns = [float(item) for item in group.get_column("net_return").drop_nulls().to_list()]
        rows.append(
            {
                "run_id": summary_row.get("run_id"),
                "regime_type": regime_type,
                "regime_value": str(value),
                "event_count": len(returns),
                "total_return": compound_values(returns),
                "mean_return": sum(returns) / len(returns) if returns else None,
                "positive_event_share": positive_share(returns),
            }
        )
    return rows


def clean_returns(events: pl.DataFrame, iterations: int) -> list[float]:
    if iterations <= 0 or events.height == 0 or "net_return" not in events.columns:
        return []
    return [float(value) for value in events.get_column("net_return").drop_nulls().to_list()]


def positive_share(values: list[float]) -> float | None:
    return sum(1 for item in values if item > 0.0) / len(values) if values else None


def negative_oos_share(walk_forward: pl.DataFrame) -> float | None:
    values = float_values(walk_forward, "oos_total_return")
    return sum(1 for value in values if value < 0.0) / len(values) if values else None


def oos_degradation_warning(
    best_is: dict[str, Any],
    config: ResearchValidationConfig,
) -> bool:
    ratio = optional_float(best_is.get("oos_to_is_return_ratio"))
    oos_return = optional_float(best_is.get("oos_total_return"))
    return (ratio is not None and ratio < config.min_best_oos_to_is_return_ratio) or (
        oos_return is not None and oos_return < config.min_best_oos_total_return
    )
