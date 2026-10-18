from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
from test_intraday_event_workflow_support import write_intraday_dataset

from backtest_engine.intraday.event.batch.config import IntradayEventBatchConfig
from backtest_engine.intraday.event.batch.runner import run_intraday_event_batch


def test_parameter_grid_uses_isolated_worker_processes(tmp_path: Path) -> None:
    plugin_path = tmp_path / "worker_pid_strategy.py"
    plugin_path.write_text(
        "import os\n\n"
        "def generate_signal(ctx, params=None):\n"
        "    return {'target': 0.0, 'features': {'worker_pid': os.getpid()}}\n",
        encoding="utf-8",
    )
    result = run_intraday_event_batch(
        IntradayEventBatchConfig(
            experiment_id="parallel_workers",
            output_root=tmp_path / "batch",
            dataset_paths=[write_intraday_dataset(tmp_path)],
            quality_approval_required=False,
            allow_legacy_bar_timestamps=True,
            strategy_file=plugin_path,
            timing="signal_close_to_same_close",
            allow_noncausal_same_close=True,
            end_of_data_position_policy="mark",
            grid={"case": [1, 2]},
            max_workers=2,
        )
    )

    worker_pids = set()
    for row in result.summary_rows:
        events = pl.read_parquet(Path(row["run_path"]) / "events.parquet")
        features = json.loads(events.get_column("strategy_features")[0])
        worker_pids.add(features["worker_pid"])

    assert len(worker_pids) == 2
    assert os.getpid() not in worker_pids
