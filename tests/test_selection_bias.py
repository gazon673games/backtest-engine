from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from backtest_engine.research.selection_bias import (
    analyze_selection_bias,
    probability_of_backtest_overfitting,
    reality_check_p_value,
)


def test_selection_bias_metrics_detect_unstable_parameter_search() -> None:
    timestamps = [datetime(2024, 1, 1) + timedelta(days=index) for index in range(40)]
    runs = []
    for run_index in range(4):
        returns = [
            (0.02 if (index + run_index) % 7 == 0 else -0.003 * (run_index + 1))
            for index in range(40)
        ]
        runs.append(
            (
                {"run_id": f"run-{run_index}"},
                pl.DataFrame({"ts": timestamps, "net_return": returns}),
            )
        )

    analysis = analyze_selection_bias(
        runs,
        bootstrap_iterations=100,
        seed=42,
        pbo_partitions=8,
    )
    summary = analysis.summary.row(0, named=True)

    assert summary["trial_count"] == 4
    assert summary["pbo_combination_count"] == 70
    assert 0.0 <= summary["probability_of_backtest_overfitting"] <= 1.0
    assert 0.0 < summary["reality_check_p_value"] <= 1.0
    assert analysis.deflated_sharpe.height == 4


def test_pbo_is_zero_when_the_same_trial_wins_every_train_and_test_split() -> None:
    probability, combinations = probability_of_backtest_overfitting(
        [[0.01, 0.02, 0.03, 0.04], [-0.01, -0.02, -0.03, -0.04]], 4
    )

    assert combinations == 6
    assert probability == 0.0


def test_pbo_matches_four_losing_selections_out_of_six_splits() -> None:
    probability, combinations = probability_of_backtest_overfitting(
        [[0.02, 0.01, -0.02, -0.01], [-0.02, -0.01, 0.02, 0.01]], 4
    )

    # При двух симметричных нулевых средних выбор и ранжирование дают ничью.
    # В остальных четырёх разбиениях победитель train проигрывает на test.
    assert combinations == 6
    assert probability == pytest.approx(2 / 3)


@pytest.mark.parametrize("constant_return,expected", [(0.01, 1 / 21), (-0.01, 1.0), (0.0, 1.0)])
def test_reality_check_constant_returns_have_an_exact_probability(
    constant_return: float, expected: float
) -> None:
    probability = reality_check_p_value([[constant_return] * 4], iterations=20, seed=42)

    assert probability == pytest.approx(expected)
