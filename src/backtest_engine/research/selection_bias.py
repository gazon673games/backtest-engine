"""Multiple-testing controls for parameter searches."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import ReportingError
from backtest_engine.research.performance import excess_kurtosis, sharpe_ratio, skewness
from backtest_engine.research.performance import mean as mean
from backtest_engine.research.performance import sample_std as sample_std
from backtest_engine.research.return_inputs import validate_return_events

RunEvents = tuple[dict[str, Any], pl.DataFrame]


@dataclass(frozen=True)
class SelectionBiasAnalysis:
    summary: pl.DataFrame
    deflated_sharpe: pl.DataFrame


def analyze_selection_bias(
    runs: list[RunEvents],
    *,
    bootstrap_iterations: int = 500,
    seed: int = 1,
    pbo_partitions: int = 8,
) -> SelectionBiasAnalysis:
    run_ids, matrix = aligned_return_matrix(runs)
    if not matrix or not matrix[0]:
        return SelectionBiasAnalysis(pl.DataFrame(), pl.DataFrame())
    trial_count = len(matrix)
    observation_count = len(matrix[0])
    sharpes = [sharpe_ratio(values) or 0.0 for values in matrix]
    benchmark = expected_maximum_sharpe(sharpes, trial_count)
    deflated_rows = [
        deflated_sharpe_row(run_id, values, sharpe, benchmark, trial_count)
        for run_id, values, sharpe in zip(run_ids, matrix, sharpes, strict=True)
    ]
    best = max(deflated_rows, key=lambda row: (row["sharpe"], str(row["run_id"])))
    pbo, combination_count = probability_of_backtest_overfitting(matrix, pbo_partitions)
    reality_p = reality_check_p_value(matrix, bootstrap_iterations, seed)
    summary = pl.DataFrame(
        [
            {
                "trial_count": trial_count,
                "observation_count": observation_count,
                "best_run_id": best["run_id"],
                "best_periodic_sharpe": best["sharpe"],
                "expected_maximum_sharpe": benchmark,
                "best_deflated_sharpe_probability": best["deflated_sharpe_probability"],
                "probability_of_backtest_overfitting": pbo,
                "pbo_combination_count": combination_count,
                "reality_check_p_value": reality_p,
                "bootstrap_iterations": bootstrap_iterations,
            }
        ],
        orient="row",
    )
    return SelectionBiasAnalysis(summary, pl.DataFrame(deflated_rows, orient="row"))


def aligned_return_matrix(runs: list[RunEvents]) -> tuple[list[str], list[list[float]]]:
    mappings: list[tuple[str, dict[Any, float]]] = []
    for summary, events in runs:
        run_id = str(summary.get("run_id"))
        validate_return_events(events, run_id)
        if any(existing == run_id for existing, _ in mappings):
            raise ReportingError(f"Duplicate research run_id: {run_id}")
        mapping = {
            row["ts"]: float(row["net_return"])
            for row in events.select("ts", "net_return").drop_nulls().iter_rows(named=True)
        }
        if mapping:
            mappings.append((str(summary.get("run_id")), mapping))
    if not mappings:
        return [], []
    timestamps = set(mappings[0][1])
    for _run_id, mapping in mappings[1:]:
        timestamps.intersection_update(mapping)
    if any(len(mapping) != len(timestamps) for _, mapping in mappings):
        raise ReportingError(
            "Selection-bias trials require identical observation timestamps; "
            "select a common evaluation interval explicitly"
        )
    ordered = sorted(timestamps)
    return (
        [run_id for run_id, _mapping in mappings],
        [[mapping[ts] for ts in ordered] for _run_id, mapping in mappings],
    )


def deflated_sharpe_row(
    run_id: str,
    returns: list[float],
    sharpe: float,
    benchmark: float,
    trial_count: int,
) -> dict[str, Any]:
    count = len(returns)
    skew = skewness(returns) or 0.0
    kurtosis = (excess_kurtosis(returns) or 0.0) + 3.0
    denominator = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    z_score = (
        0.0
        if count < 2 or denominator <= 0.0
        else (sharpe - benchmark) * math.sqrt(count - 1) / math.sqrt(denominator)
    )
    return {
        "run_id": run_id,
        "trial_count": trial_count,
        "observation_count": count,
        "sharpe": sharpe,
        "expected_maximum_sharpe": benchmark,
        "deflated_sharpe_z": z_score,
        "deflated_sharpe_probability": normal_cdf(z_score),
        "return_skewness": skew,
        "return_kurtosis": kurtosis,
    }


def expected_maximum_sharpe(sharpes: list[float], trial_count: int) -> float:
    if trial_count <= 1:
        return 0.0
    deviation = sample_std(sharpes)
    if deviation == 0.0:
        return 0.0
    euler_gamma = 0.5772156649015329
    first = normal_ppf(1.0 - 1.0 / trial_count)
    second = normal_ppf(1.0 - math.exp(-1.0) / trial_count)
    return deviation * ((1.0 - euler_gamma) * first + euler_gamma * second)


def probability_of_backtest_overfitting(
    matrix: list[list[float]], partitions: int
) -> tuple[float | None, int]:
    trial_count = len(matrix)
    observations = len(matrix[0])
    partitions = min(partitions, observations)
    if trial_count < 2 or partitions < 4:
        return None, 0
    if partitions % 2:
        partitions -= 1
    slices = partition_indices(observations, partitions)
    combinations = itertools.combinations(range(partitions), partitions // 2)
    overfit = 0
    evaluated = 0
    for selected_partitions in combinations:
        # Evaluate a split and its complement together, retaining both original observations.
        if selected_partitions[0] != 0:
            continue
        selected_set = set(selected_partitions)
        complement = tuple(part for part in range(partitions) if part not in selected_set)
        train_indices = [index for part in selected_set for index in slices[part]]
        test_indices = [
            index
            for part in range(partitions)
            if part not in selected_set
            for index in slices[part]
        ]
        train_scores = [subset_sharpe(values, train_indices) for values in matrix]
        test_scores = [subset_sharpe(values, test_indices) for values in matrix]
        overfit += overfit_selection(train_scores, test_scores)
        # Preserve the old reduction order even if set iteration differs from sorted partitions.
        reverse_train = test_scores
        if tuple(set(complement)) != complement:
            indices = [index for part in set(complement) for index in slices[part]]
            reverse_train = [subset_sharpe(values, indices) for values in matrix]
        reverse_test = train_scores
        if tuple(selected_set) != selected_partitions:
            indices = [index for part in selected_partitions for index in slices[part]]
            reverse_test = [subset_sharpe(values, indices) for values in matrix]
        overfit += overfit_selection(reverse_train, reverse_test)
        evaluated += 2
    return (overfit / evaluated if evaluated else None), evaluated


def overfit_selection(train_scores: list[float], test_scores: list[float]) -> int:
    selected = max(range(len(train_scores)), key=lambda index: (train_scores[index], index))
    rank = sorted(range(len(test_scores)), key=lambda index: test_scores[index]).index(selected) + 1
    return int(rank / (len(train_scores) + 1.0) <= 0.5)


def reality_check_p_value(matrix: list[list[float]], iterations: int, seed: int) -> float | None:
    if iterations <= 0 or not matrix or not matrix[0]:
        return None
    count = len(matrix[0])
    observed = max(math.sqrt(count) * mean(values) for values in matrix)
    centered = []
    for values in matrix:
        average = mean(values)
        centered.append([value - average for value in values])
    rng = random.Random(seed)
    scale = math.sqrt(count)
    exceedances = 0
    for _ in range(iterations):
        indices = [rng.randrange(count) for _ in range(count)]
        statistic = max(
            scale * (sum(map(values.__getitem__, indices)) / count) for values in centered
        )
        exceedances += int(statistic >= observed)
    return (exceedances + 1) / (iterations + 1)


def partition_indices(count: int, partitions: int) -> list[list[int]]:
    return [
        list(
            range(
                math.floor(part * count / partitions),
                math.floor((part + 1) * count / partitions),
            )
        )
        for part in range(partitions)
    ]


def subset_sharpe(values: list[float], indices: list[int]) -> float:
    return sharpe_ratio([values[index] for index in indices]) or 0.0


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_ppf(probability: float) -> float:
    low, high = -10.0, 10.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if normal_cdf(midpoint) < probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0
