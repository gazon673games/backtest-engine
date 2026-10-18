"""Bounded portfolio views from the reconciled native ledger and confirmed fills."""

from math import ceil
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.data.store import file_sha256
from backtest_engine.workbench._report_builder import (
    equity_frame,
    read_candle_table,
    read_json,
    read_table,
)
from backtest_engine.workbench._report_frames import frame_records, table_records
from backtest_engine.workbench.preview_storage import downsample_ohlcv


def build_portfolio_report(output_dir: Path, *, max_points: int = 5000) -> dict[str, Any]:
    config = read_json(output_dir / "config.json")
    ledger = read_table(output_dir / "portfolio_ledger.parquet")
    legs = read_table(output_dir / "portfolio_legs.parquet")
    initial = config["portfolio"]["initial_capital"]
    manifest = read_json(output_dir / "workbench_portfolio.json")
    assets = []
    for leg in config["legs"]:
        asset = leg["asset_id"]
        rows = legs.filter(pl.col("asset_id") == asset)
        fills = rows.filter(pl.col("filled_contract_delta").abs() > 0)
        path = Path(leg["dataset_path"])
        expected = manifest.get("input_pins", {}).get(str(path.resolve()))
        warning = None
        if not path.exists():
            warning = f"Chart dataset missing: {path}"
        elif expected is not None and file_sha256(path) != expected:
            warning = f"Chart dataset changed since this run: {path}"
        candles = pl.DataFrame() if warning else read_candle_table(path)
        if ledger.height and candles.height:
            start, end = rows["ts"].min(), ledger["ts"].max()
            dtype = candles.schema["ts"]
            candles = candles.filter(
                pl.col("ts").is_between(
                    pl.lit(start).cast(dtype),
                    pl.lit(end).cast(dtype),
                )
            )
        stride = max(1, ceil(candles.height / max(1, max_points)))
        assets.append(
            {
                "asset_id": asset,
                "dataset_path": str(path),
                "dataset_id": leg.get("dataset_id"),
                "candles": table_records(downsample_ohlcv(candles, stride))
                if candles.height
                else [],
                "candle_stride": stride,
                "chart_warning": warning,
                "position": frame_records(
                    rows.select(
                        pl.col("next_ts").alias("ts"),
                        "target_contracts",
                        "position_notional",
                    ),
                    max_points,
                    transition_columns=("target_contracts",),
                ),
                "pnl": frame_records(
                    rows.select(
                        pl.col("next_ts").alias("ts"),
                        pl.col("net_pnl_base").cum_sum().alias("net_pnl"),
                    ),
                    max_points,
                ),
                "fills": table_records(fills.tail(2000)),
                "fill_count": fills.height,
            }
        )
    return {
        "kind": "portfolio",
        "run_id": output_dir.name,
        "output_dir": str(output_dir),
        "config": config,
        "metrics": read_json(output_dir / "metrics.json"),
        "manifest": manifest,
        "quality_approval": read_json(output_dir / "quality_approvals.json"),
        "dataset_validation": read_json(output_dir / "dataset_validations.json"),
        "assets": assets,
        "series": {
            "equity": frame_records(
                equity_frame(ledger, initial), max_points, extrema_column="drawdown"
            )
        },
        "tables": {"trades": [], "decision_actions": []},
    }
