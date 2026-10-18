from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.research.selection_bias import (
    probability_of_backtest_overfitting,
    reality_check_p_value,
)
from backtest_engine.research.validation.analytics import block_bootstrap_metrics, bootstrap_metrics
from backtest_engine.research.validation.config import ResearchValidationConfig

# Captured before optimization; includes zero equity, negative factors, ties and near-zero variance.
GOLDEN = json.loads((Path(__file__).parent / "fixtures/research_sampling_golden.json").read_text())


@pytest.mark.parametrize("case", GOLDEN["sampling"])
def test_bootstrap_keeps_exact_seeded_results_and_partial_blocks(case: dict) -> None:
    config = ResearchValidationConfig(
        experiment_path=Path("fixture"),
        bootstrap_iterations=31,
        block_bootstrap_iterations=29,
        bootstrap_seed=case["seed"],
        block_bootstrap_length=case["block_length"],
    )
    events = pl.DataFrame({"net_return": case["returns"]})
    assert bootstrap_metrics({"run_id": "fixture"}, events, config, 3) == case["bootstrap"]
    assert (
        block_bootstrap_metrics({"run_id": "fixture"}, events, config, 3) == case["block_bootstrap"]
    )


@pytest.mark.parametrize("case", GOLDEN["pbo"])
def test_pbo_ties_odd_partitions_and_reality_check_keep_exact_results(case: dict) -> None:
    assert probability_of_backtest_overfitting(case["matrix"], case["partitions"]) == tuple(
        case["expected"]
    )
    assert reality_check_p_value(case["matrix"], 31, 42) == case["reality"]


@pytest.mark.parametrize("iterations", [0, 10])
@pytest.mark.parametrize("values", [[], [None, None]])
def test_disabled_or_empty_bootstrap_returns_no_metrics(iterations: int, values: list) -> None:
    config = ResearchValidationConfig(
        experiment_path=Path("fixture"),
        bootstrap_iterations=iterations,
        block_bootstrap_iterations=iterations,
    )
    events = pl.DataFrame({"net_return": values}, schema={"net_return": pl.Float64})
    assert bootstrap_metrics({}, events, config, 0) is None
    assert block_bootstrap_metrics({}, events, config, 0) is None


def test_disabled_bootstrap_with_available_returns_has_no_metrics() -> None:
    config = ResearchValidationConfig(
        experiment_path=Path("fixture"),
        bootstrap_iterations=0,
        block_bootstrap_iterations=0,
    )
    events = pl.DataFrame({"net_return": [0.01]})
    assert bootstrap_metrics({}, events, config, 0) is None
    assert block_bootstrap_metrics({}, events, config, 0) is None


def test_pbo_insufficient_history_and_disabled_reality_check() -> None:
    assert probability_of_backtest_overfitting([[0.1], [0.2]], 8) == (None, 0)
    assert probability_of_backtest_overfitting([[0.1] * 8], 8) == (None, 0)
    assert reality_check_p_value([[0.1]], 0, 42) is None
