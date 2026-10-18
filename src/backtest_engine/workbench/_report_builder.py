from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.reporting.nav import drawdown_values, initial_nav
from backtest_engine.workbench._report_frames import (
    cost_frame,
    extract_plot_series,
    frame_records,
    selected_frame,
    table_records,
)

REPORT_TABLES = ("decision_actions", "trades")
EVENT_REPORT_COLUMNS = (
    "ts",
    "next_ts",
    "fill_ts",
    "target_position",
    "target_contracts",
    "gross_pnl",
    "net_pnl",
    "net_return",
    "cost_cash",
    "commission_cost_cash",
    "slippage_cost_cash",
    "spread_cost_cash",
    "roll_cost_cash",
    "strategy_features",
)
EQUITY_REPORT_COLUMNS = ("ts", "next_ts", "equity", "equity_cash", "net_pnl")


def build_workbench_report(
    output_dir: Path,
    *,
    dataset_path: Path | None = None,
    max_points: int = 5000,
) -> dict[str, Any]:
    config = read_json(output_dir / "config.json")
    if "legs" in config:
        from backtest_engine.workbench.portfolio_report import build_portfolio_report

        return build_portfolio_report(output_dir, max_points=max_points)
    quality_dataset = dataset_path
    if quality_dataset is None and config.get("dataset", {}).get("path"):
        quality_dataset = Path(config["dataset"]["path"])
    events = read_table(output_dir / "events.parquet", columns=EVENT_REPORT_COLUMNS)
    equity = read_table(output_dir / "equity.parquet", columns=EQUITY_REPORT_COLUMNS)
    decisions = read_table(output_dir / "decision_actions.parquet")
    trades = read_table(output_dir / "trades.parquet")
    dataset = (
        read_candle_table(dataset_path)
        if dataset_path is not None and dataset_path.exists()
        else None
    )
    return {
        "run_id": output_dir.name,
        "output_dir": str(output_dir),
        "metrics": read_json(output_dir / "metrics.json"),
        "config": config,
        "dataset_validation": read_json(output_dir / "dataset_validation.json"),
        "quality_approval": read_json(output_dir / "quality_approval.json"),
        "dataset_quality": read_dataset_quality(quality_dataset),
        "series": {
            "candles": frame_records(candle_frame(dataset), max_points),
            "equity": frame_records(equity_frame(equity), max_points, extrema_column="drawdown"),
            "position": frame_records(
                position_frame(events),
                max_points,
                transition_columns=("target_position", "target_contracts"),
            ),
            "pnl": frame_records(pnl_frame(events), max_points),
            "costs": frame_records(cost_frame(events), max_points),
            "plots": extract_plot_series(events, max_points=max_points),
        },
        "tables": {
            "decision_actions": table_records(decisions),
            "trades": table_records(trades),
        },
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(
    path: Path | None,
    *,
    columns: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    if path is None or not path.exists():
        return pl.DataFrame()
    if path.suffix == ".parquet":
        if columns is None:
            return pl.read_parquet(path)
        schema = pl.read_parquet_schema(path)
        projected = [column for column in columns if column in schema]
        return pl.scan_parquet(path).select(projected).collect()
    if path.suffix == ".csv":
        return pl.read_csv(path)
    return pl.DataFrame()


def read_candle_table(path: Path) -> pl.DataFrame:
    if path.suffix != ".parquet":
        return read_table(path)
    schema = pl.read_parquet_schema(path)
    columns = [name for name in ("ts", "open", "high", "low", "close", "volume") if name in schema]
    return pl.read_parquet(path, columns=columns)


def read_dataset_quality(dataset_path: Path | None) -> dict[str, Any]:
    if dataset_path is None:
        return {}
    candidates = [
        dataset_path.parent / "quality_approval.json",
        dataset_path.parent.parent / "reports" / "approval.json",
        dataset_path.parent / "reports" / "approval.json",
    ]
    for path in candidates:
        payload = read_json(path)
        if payload:
            return payload
    return {}


def candle_frame(frame: pl.DataFrame | None) -> pl.DataFrame:
    if frame is None or frame.height == 0:
        return pl.DataFrame()
    columns = [
        name for name in ("ts", "open", "high", "low", "close", "volume") if name in frame.columns
    ]
    return frame.select(columns)


def equity_frame(frame: pl.DataFrame, initial_capital: float | None = None) -> pl.DataFrame:
    if frame.height == 0:
        return frame
    result = selected_frame(frame, ("equity", "equity_cash", "net_pnl"), timestamp_column="next_ts")
    equity_column = "equity_cash" if "equity_cash" in result.columns else "equity"
    if equity_column in result.columns:
        values = [float(value) for value in result[equity_column]]
        result = result.with_columns(
            pl.Series(
                "drawdown",
                drawdown_values(values, initial_nav(frame, equity_column, initial_capital)),
            )
        )
    return result


def position_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return selected_frame(
        frame, ("target_position", "target_contracts"), timestamp_column="fill_ts"
    )


def pnl_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return selected_frame(frame, ("gross_pnl", "net_pnl", "net_return"), timestamp_column="next_ts")
