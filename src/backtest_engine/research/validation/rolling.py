"""Leakage-resistant parameter selection across rolling folds and a locked holdout."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import polars as pl

from backtest_engine.research.validation.config import ResearchValidationConfig
from backtest_engine.research.validation.stats import segment_metrics

RunEvents = tuple[dict[str, Any], pl.DataFrame]


def rolling_validation(
    runs: list[RunEvents],
    config: ResearchValidationConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    dates = observed_dates(runs)
    if len(dates) < 4 or not runs:
        return pl.DataFrame(), pl.DataFrame(), ["rolling validation needs at least four dates"]
    holdout_start = resolve_holdout_start(dates, config)
    development_dates = [value for value in dates if value < holdout_start]
    folds, warnings = rolling_fold_rows(runs, development_dates, config)
    locked = locked_holdout_rows(runs, dates, holdout_start, config)
    if locked.height == 0:
        warnings.append("locked holdout has no evaluable candidate")
    return folds, locked, warnings


def observed_dates(runs: list[RunEvents]) -> list[date]:
    result: set[date] = set()
    for _summary, events in runs:
        if "ts" not in events.columns:
            continue
        result.update(as_date(value) for value in events.get_column("ts").to_list())
    return sorted(result)


def resolve_holdout_start(dates: list[date], config: ResearchValidationConfig) -> date:
    explicit = config.locked_holdout_start or config.split_date
    if explicit is not None:
        return explicit
    holdout_index = math.floor(len(dates) * (1 - config.locked_holdout_fraction))
    index = max(1, min(len(dates) - 1, holdout_index))
    return dates[index]


def rolling_fold_rows(
    runs: list[RunEvents],
    dates: list[date],
    config: ResearchValidationConfig,
) -> tuple[pl.DataFrame, list[str]]:
    warnings: list[str] = []
    minimum = min(config.rolling_min_train_dates, max(2, len(dates) // 2))
    remaining = len(dates) - minimum
    fold_count = min(config.rolling_folds, remaining)
    if fold_count < 1:
        return pl.DataFrame(), ["not enough development dates for a rolling fold"]
    rows = []
    for fold in range(fold_count):
        test_start_index = minimum + math.floor(fold * remaining / fold_count)
        test_end_index = minimum + math.floor((fold + 1) * remaining / fold_count)
        row = evaluate_fold(runs, dates, test_start_index, test_end_index, fold + 1, config)
        if row is None:
            warnings.append(f"rolling fold {fold + 1} has no evaluable candidate")
        else:
            rows.append(row)
    return frame(rows), warnings


def evaluate_fold(
    runs: list[RunEvents],
    dates: list[date],
    test_start_index: int,
    test_end_index: int,
    fold: int,
    config: ResearchValidationConfig,
) -> dict[str, Any] | None:
    exclusion = config.purge_dates + config.embargo_dates
    train_end_index = max(0, test_start_index - exclusion)
    train_start_index = 0
    if config.rolling_train_dates is not None:
        train_start_index = max(0, train_end_index - config.rolling_train_dates)
    if train_end_index <= train_start_index or test_end_index <= test_start_index:
        return None
    train_start, train_end = dates[train_start_index], dates[train_end_index - 1]
    test_start, test_end = dates[test_start_index], dates[test_end_index - 1]
    selected = select_candidate(runs, train_start, train_end, config.selection_metric)
    if selected is None:
        return None
    summary, events, score = selected
    test = date_slice(events, test_start, test_end)
    if test.height == 0:
        return None
    return {
        "fold": fold,
        "selected_run_id": summary.get("run_id"),
        "selection_metric": config.selection_metric,
        "selection_score": score,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "purge_dates": config.purge_dates,
        "embargo_dates": config.embargo_dates,
        **segment_metrics("train", date_slice(events, train_start, train_end)),
        **segment_metrics("test", test),
    }


def locked_holdout_rows(
    runs: list[RunEvents],
    dates: list[date],
    holdout_start: date,
    config: ResearchValidationConfig,
) -> pl.DataFrame:
    development = [value for value in dates if value < holdout_start]
    holdout = [value for value in dates if value >= holdout_start]
    exclusion = config.purge_dates + config.embargo_dates
    if len(development) <= exclusion or not holdout:
        return pl.DataFrame()
    train_end = development[-1 - exclusion]
    selected = select_candidate(runs, development[0], train_end, config.selection_metric)
    if selected is None:
        return pl.DataFrame()
    summary, events, score = selected
    holdout_events = date_slice(events, holdout[0], holdout[-1])
    if holdout_events.height == 0:
        return pl.DataFrame()
    row = {
        "selected_run_id": summary.get("run_id"),
        "selection_metric": config.selection_metric,
        "selection_score": score,
        "locked": True,
        "train_start": development[0].isoformat(),
        "train_end": train_end.isoformat(),
        "holdout_start": holdout[0].isoformat(),
        "holdout_end": holdout[-1].isoformat(),
        **segment_metrics("train", date_slice(events, development[0], train_end)),
        **segment_metrics("holdout", holdout_events),
    }
    return frame([row])


def select_candidate(
    runs: list[RunEvents],
    start: date,
    end: date,
    metric: str,
) -> tuple[dict[str, Any], pl.DataFrame, float] | None:
    candidates = []
    for summary, events in runs:
        sample = date_slice(events, start, end)
        score = selection_score(sample, metric)
        if score is not None:
            candidates.append((summary, events, score))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[2], str(item[0].get("run_id"))))


def selection_score(events: pl.DataFrame, metric: str) -> float | None:
    if events.height == 0 or "net_return" not in events.columns:
        return None
    values = [float(value) for value in events.get_column("net_return").drop_nulls().to_list()]
    if not values:
        return None
    if metric == "total_return":
        equity = math.prod(1.0 + value for value in values)
        return equity - 1.0
    if metric == "sharpe":
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return None if variance <= 0.0 else mean / math.sqrt(variance) * math.sqrt(len(values))
    raise ValueError(f"Unsupported rolling selection_metric: {metric}")


def date_slice(events: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    return events.filter(pl.col("ts").dt.date().is_between(start, end, closed="both"))


def as_date(value: Any) -> date:
    return value.date() if isinstance(value, datetime) else value


def frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, orient="row") if rows else pl.DataFrame()
