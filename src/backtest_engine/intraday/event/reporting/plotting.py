from __future__ import annotations

# ruff: noqa: E402, I001

import math
from pathlib import Path
from typing import Any

import matplotlib
import polars as pl

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_decision_window_heatmap(
    summary: pl.DataFrame,
    metric: str,
    output_path: Path,
    *,
    title: str,
) -> Path:
    require_columns(summary, {"decision_time", "window_minutes", metric})
    rows = [row for row in summary.iter_rows(named=True) if row.get(metric) is not None]
    if not rows:
        raise ValueError(f"no non-null {metric} values")
    decision_times, windows = heatmap_axes(rows)
    matrix = heatmap_matrix(rows, decision_times, windows, metric)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(decision_times)), labels=decision_times)
    ax.set_yticks(range(len(windows)), labels=[compact_number(value) for value in windows])
    ax.set_xlabel("decision_time")
    ax.set_ylabel("window_minutes")
    ax.set_title(title)
    annotate_matrix(ax, matrix)
    fig.colorbar(image, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_cost_stress(
    summary: pl.DataFrame,
    metric: str,
    output_path: Path,
    *,
    title: str,
) -> Path:
    require_columns(summary, {"commission_bps", "slippage_bps", metric})
    best_by_cost: dict[float, float] = {}
    for row in summary.iter_rows(named=True):
        value = row.get(metric)
        if value is None:
            continue
        cost = float(row["commission_bps"]) + float(row["slippage_bps"])
        best_by_cost[cost] = max(float(value), best_by_cost.get(cost, float("-inf")))
    if not best_by_cost:
        raise ValueError(f"no non-null {metric} values")
    x_values = sorted(best_by_cost)
    y_values = [best_by_cost[value] for value in x_values]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(x_values, y_values, marker="o", linewidth=1.5)
    ax.set_xlabel("commission_bps + slippage_bps")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_long_short_exposure(top_runs: pl.DataFrame, output_path: Path, *, title: str) -> Path:
    require_columns(top_runs, {"run_id", "long_exposure", "short_exposure"})
    rows = list(top_runs.head(10).iter_rows(named=True))
    if not rows:
        raise ValueError("top_runs is empty")
    labels = [str(row["run_id"])[:18] for row in rows]
    long_values = [float(row["long_exposure"] or 0.0) for row in rows]
    short_values = [float(row["short_exposure"] or 0.0) for row in rows]
    x_values = list(range(len(rows)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.bar(x_values, long_values, label="long_exposure")
    ax.bar(x_values, short_values, bottom=long_values, label="short_exposure")
    ax.set_xticks(x_values, labels=labels, rotation=35, ha="right")
    ax.set_ylabel("Exposure")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def require_columns(frame: pl.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    if frame.height == 0:
        raise ValueError("summary is empty")


def heatmap_axes(rows: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
    decision_times = sorted({str(row["decision_time"]) for row in rows if row["decision_time"]})
    raw_windows = [as_float(row["window_minutes"]) for row in rows if row["window_minutes"]]
    windows = sorted({value for value in raw_windows if value is not None})
    if not decision_times or not windows:
        raise ValueError("decision_time and window_minutes values are required")
    return decision_times, windows


def heatmap_matrix(
    rows: list[dict[str, Any]],
    decision_times: list[str],
    windows: list[float],
    metric: str,
) -> list[list[float]]:
    matrix = [[math.nan for _ in decision_times] for _ in windows]
    window_index = {value: index for index, value in enumerate(windows)}
    time_index = {value: index for index, value in enumerate(decision_times)}
    for row in rows:
        decision_time = str(row.get("decision_time"))
        window = as_float(row.get("window_minutes"))
        value = row.get(metric)
        if decision_time not in time_index or window not in window_index or value is None:
            continue
        y_index = window_index[window]
        x_index = time_index[decision_time]
        value_float = float(value)
        if not math.isfinite(matrix[y_index][x_index]) or value_float > matrix[y_index][x_index]:
            matrix[y_index][x_index] = value_float
    return matrix


def annotate_matrix(ax: Any, matrix: list[list[float]]) -> None:
    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            if math.isfinite(value):
                ax.text(x_index, y_index, f"{value:.2f}", ha="center", va="center", color="white")


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compact_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"
