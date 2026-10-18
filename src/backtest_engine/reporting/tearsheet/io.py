from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.research.validation.io import write_table as write_table


def read_optional_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def render_report(
    *,
    run_path: Path,
    events: pl.DataFrame,
    trades: pl.DataFrame,
    monthly_returns: pl.DataFrame,
    drawdown_periods: pl.DataFrame,
    cost_breakdown: pl.DataFrame,
    trade_distribution: pl.DataFrame,
) -> str:
    lines = [
        "# Intraday Event Tearsheet",
        "",
        f"- run_path: `{run_path}`",
        f"- event_count: {events.height}",
        f"- trade_count: {trades.height}",
        f"- total_return: {_format_float(_last(events, 'equity', default=1.0) - 1.0)}",
        f"- final_equity_cash: {_format_float(_last(events, 'equity_cash'))}",
        "",
        "## Monthly Returns",
        "",
        markdown_table(monthly_returns),
        "",
        "## Drawdown Periods",
        "",
        markdown_table(drawdown_periods),
        "",
        "## Cost Breakdown",
        "",
        markdown_table(cost_breakdown),
        "",
        "## Trade Distribution",
        "",
        markdown_table(trade_distribution),
        "",
    ]
    return "\n".join(lines)


def markdown_table(frame: pl.DataFrame) -> str:
    if frame.height == 0 or not frame.columns:
        return "_No rows._"
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    lines = [header, divider]
    lines.extend(
        "| " + " | ".join(_format_float(value) for value in row) + " |" for row in frame.iter_rows()
    )
    return "\n".join(lines)


def _last(frame: pl.DataFrame, column: str, default: float | None = None) -> float | None:
    if frame.height == 0 or column not in frame.columns:
        return default
    value = frame.get_column(column)[-1]
    return None if value is None else float(value)


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value).replace("|", "\\|")
