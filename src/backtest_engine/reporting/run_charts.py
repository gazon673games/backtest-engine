from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
import polars as pl

from backtest_engine.reporting.nav import drawdown_values

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_equity_curve(
    equity_path: Path,
    output_path: Path,
    *,
    title: str = "Best overall equity curve",
) -> Path:
    equity = pl.read_parquet(equity_path)
    _require_columns(equity, ["ts", "equity"], equity_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(equity.get_column("ts").to_list(), equity.get_column("equity").to_list(), linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_drawdown_curve(
    equity_path: Path,
    output_path: Path,
    *,
    title: str = "Best overall drawdown",
) -> Path:
    equity = pl.read_parquet(equity_path)
    _require_columns(equity, ["ts", "equity"], equity_path)
    equity_values = [float(value) for value in equity.get_column("equity").to_list()]
    drawdowns = drawdown_values(equity_values, 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(equity.get_column("ts").to_list(), drawdowns, linewidth=1.5, color="#b3261e")
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_parameter_heatmap(
    summary: pl.DataFrame,
    dataset_symbol: str,
    metric: str,
    output_path: Path,
) -> Path:
    required = {"continuous_symbol", "fast_window", "slow_window", metric}
    missing = sorted(required.difference(summary.columns))
    if missing:
        msg = f"Cannot plot heatmap, missing columns: {', '.join(missing)}"
        raise ValueError(msg)

    dataset_summary = summary.filter(pl.col("continuous_symbol") == dataset_symbol)
    if dataset_summary.height == 0:
        msg = f"Cannot plot heatmap, no rows for dataset: {dataset_symbol}"
        raise ValueError(msg)

    fast_values = sorted(int(value) for value in dataset_summary.get_column("fast_window").unique())
    slow_values = sorted(int(value) for value in dataset_summary.get_column("slow_window").unique())
    matrix = [[math.nan for _ in slow_values] for _ in fast_values]

    fast_index = {value: index for index, value in enumerate(fast_values)}
    slow_index = {value: index for index, value in enumerate(slow_values)}
    for row in dataset_summary.iter_rows(named=True):
        value = row.get(metric)
        if value is None:
            continue
        matrix[fast_index[int(row["fast_window"])]][slow_index[int(row["slow_window"])]] = float(
            value
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(slow_values)), labels=[str(value) for value in slow_values])
    ax.set_yticks(range(len(fast_values)), labels=[str(value) for value in fast_values])
    ax.set_xlabel("slow_window")
    ax.set_ylabel("fast_window")
    adjustment_mode = _single_value(dataset_summary, "adjustment_mode")
    title_suffix = f" ({adjustment_mode})" if adjustment_mode else ""
    ax.set_title(f"{dataset_symbol} {metric} heatmap{title_suffix}")

    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            if math.isfinite(value):
                ax.text(x_index, y_index, f"{value:.2f}", ha="center", va="center", color="white")

    fig.colorbar(image, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _require_columns(frame: pl.DataFrame, columns: list[str], path: Path) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        msg = f"{path} is missing required chart columns: {', '.join(missing)}"
        raise ValueError(msg)


def _single_value(frame: pl.DataFrame, column: str) -> Any:
    if column not in frame.columns:
        return None
    values = frame.get_column(column).drop_nulls().unique().to_list()
    return values[0] if len(values) == 1 else None
