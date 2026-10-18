# ruff: noqa
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from backtest_engine.cli import app
from backtest_engine.intraday.event.batch.runner import (
    IntradayEventBatchConfig,
    expand_intraday_event_grid,
    generate_intraday_event_run_id,
    load_intraday_event_batch_config,
    parse_intraday_event_grid_options,
    run_intraday_event_batch,
    split_intraday_event_params,
)
from backtest_engine.intraday.event.reporting.report import (
    IntradayEventReportConfig,
    generate_intraday_event_report,
)

runner = CliRunner()


def make_report_experiment(tmp_path: Path) -> Path:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_mode_plugin(tmp_path)
    result = run_intraday_event_batch(
        IntradayEventBatchConfig(
            experiment_id="report_test",
            output_root=tmp_path / "batch",
            dataset_paths=[dataset_path],
            quality_approval_required=False,
            allow_legacy_bar_timestamps=True,
            strategy_file=plugin_path,
            timing="signal_close_to_same_close",
            allow_noncausal_same_close=True,
            end_of_data_position_policy="mark",
            schedule="times",
            grid={
                "decision_time": ["13:00", "13:30"],
                "window_minutes": [180, 210],
                "commission_bps": [0],
                "slippage_bps": [0],
                "mode": ["long"],
            },
        )
    )
    return result.output_dir


def write_intraday_dataset(tmp_path: Path) -> Path:
    timestamps = [
        datetime(2024, 1, 2, 13, 0),
        datetime(2024, 1, 2, 13, 30),
        datetime(2024, 1, 2, 13, 35),
    ]
    closes = [100.0, 110.0, 120.0]
    dataset_path = tmp_path / "intraday.parquet"
    active_contracts = ["CRH4"] * len(timestamps)
    pl.DataFrame(
        {
            "source": ["fixture"] * len(timestamps),
            "exchange": ["MOEX"] * len(timestamps),
            "engine": ["futures"] * len(timestamps),
            "market": ["forts"] * len(timestamps),
            "board": ["RFUD"] * len(timestamps),
            "symbol": active_contracts,
            "continuous_symbol": ["CNYRUB_NEAREST_5M"] * len(timestamps),
            "timeframe": ["5m"] * len(timestamps),
            "ts": timestamps,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * len(timestamps),
            "value": [100000.0] * len(timestamps),
            "currency": ["RUB"] * len(timestamps),
            "active_contract": active_contracts,
            "source_contract_symbol": active_contracts,
            "dataset_type": ["nearest_contract"] * len(timestamps),
            "adjustment_mode": ["unadjusted"] * len(timestamps),
        }
    ).write_parquet(dataset_path)
    return dataset_path


def write_mode_plugin(tmp_path: Path) -> Path:
    plugin_path = tmp_path / "event_mode_strategy.py"
    plugin_path.write_text(
        """
def generate_signal(ctx, params=None):
    params = params or {}
    mode = params.get("mode", "long")
    if mode == "fail":
        raise ValueError("mode fail requested")
    if mode == "short":
        return -1.0
    if mode == "flat":
        return 0.0
    return 1.0
""".lstrip(),
        encoding="utf-8",
    )
    return plugin_path
