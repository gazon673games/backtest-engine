from __future__ import annotations

import tracemalloc
from pathlib import Path

import polars as pl

from backtest_engine.research.validation.analytics import block_bootstrap_metrics
from backtest_engine.research.validation.config import ResearchValidationConfig


def test_block_bootstrap_does_not_materialize_all_overlapping_blocks() -> None:
    count = 10_000
    events = pl.DataFrame({"net_return": [0.0001] * count})
    config = ResearchValidationConfig(
        experiment_path=Path("fixture"),
        block_bootstrap_iterations=2,
        block_bootstrap_length=64,
    )
    tracemalloc.start()
    try:
        result = block_bootstrap_metrics({"run_id": "fixture"}, events, config, 0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result["block_bootstrap_iterations"] == 2
    # Python allocation ceiling scales with input length, not length * overlapping block size.
    assert peak < count * 100
