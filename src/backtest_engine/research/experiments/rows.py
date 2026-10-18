from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backtest_engine.core.exceptions import ReportingError
from backtest_engine.research.experiments.config import REGISTRY_COLUMNS
from backtest_engine.research.experiments.io import file_sha256
from backtest_engine.research.experiments.utils import (
    decision_time_text,
    dict_value,
    json_text,
    normalize_path_key,
    optional_string,
)


def registry_rows(root: Path) -> tuple[list[dict[str, Any]], set[str], list[str]]:
    if not root.exists():
        msg = f"Experiment root does not exist: {root}"
        raise ReportingError(msg)

    batch_payloads: list[tuple[Path, dict[str, Any]]] = []
    run_payloads: list[tuple[Path, dict[str, Any]]] = []
    warnings: list[str] = []
    for manifest_path in sorted(root.rglob("experiment_manifest.json")):
        payload = read_manifest(manifest_path, warnings)
        if payload is None:
            continue
        if is_run_manifest(payload):
            run_payloads.append((manifest_path, payload))
        elif is_batch_manifest(payload):
            batch_payloads.append((manifest_path, payload))

    parent_by_manifest = parent_map(batch_payloads)
    experiment_ids = {
        str(payload.get("experiment_id"))
        for _path, payload in batch_payloads
        if payload.get("experiment_id")
    }
    rows = [
        run_manifest_row(manifest_path, payload, parent_by_manifest)
        for manifest_path, payload in run_payloads
    ]
    for manifest_path, payload in batch_payloads:
        rows.extend(failed_rows_from_batch_manifest(manifest_path, payload))
    return rows, experiment_ids, warnings


def read_manifest(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as manifest_file:
            payload = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Skipped unreadable manifest {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        warnings.append(f"Skipped non-object manifest {path}")
        return None
    return payload


def is_run_manifest(payload: dict[str, Any]) -> bool:
    return all(key in payload for key in ("run_id", "dataset", "strategy", "metrics"))


def is_batch_manifest(payload: dict[str, Any]) -> bool:
    return "run_manifests" in payload or "failed_runs" in payload


def parent_map(
    batch_payloads: list[tuple[Path, dict[str, Any]]],
) -> dict[str, dict[str, str | None]]:
    parents: dict[str, dict[str, str | None]] = {}
    for manifest_path, payload in batch_payloads:
        experiment_id = optional_string(payload.get("experiment_id"))
        experiment_path = optional_string(payload.get("output_dir")) or str(manifest_path.parent)
        for run_manifest in payload.get("run_manifests", []):
            if not isinstance(run_manifest, dict):
                continue
            child_manifest_path = run_manifest.get("manifest_path")
            if child_manifest_path is None:
                continue
            parents[normalize_path_key(Path(str(child_manifest_path)))] = {
                "parent_experiment_id": experiment_id,
                "parent_experiment_path": experiment_path,
            }
    return parents


def run_manifest_row(
    manifest_path: Path,
    payload: dict[str, Any],
    parent_by_manifest: dict[str, dict[str, str | None]],
) -> dict[str, Any]:
    dataset = dict_value(payload.get("dataset"))
    validation = dict_value(dataset.get("validation"))
    strategy = dict_value(payload.get("strategy"))
    metrics = dict_value(payload.get("metrics"))
    execution = dict_value(payload.get("execution"))
    portfolio = dict_value(payload.get("portfolio"))
    quality_approval = dict_value(payload.get("quality_approval"))
    catalog_approval = dict_value(payload.get("catalog_approval"))
    parent = parent_by_manifest.get(normalize_path_key(manifest_path), {})

    return ordered_registry_row(
        {
            "status": "success",
            "parent_experiment_id": parent.get("parent_experiment_id"),
            "parent_experiment_path": parent.get("parent_experiment_path"),
            "run_id": payload.get("run_id"),
            "run_path": str(manifest_path.parent),
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "dataset_path": dataset.get("path"),
            "dataset_sha256": dataset.get("sha256"),
            "dataset_version_id": dataset.get("version_id"),
            "dataset_schema_hash": dataset.get("schema_hash"),
            "dataset_schema_version": dataset.get("schema_version"),
            "continuous_symbol": validation.get("continuous_symbol"),
            "timeframe": validation.get("timeframe"),
            "dataset_type": validation.get("dataset_type"),
            "adjustment_mode": validation.get("adjustment_mode"),
            "quality_approval_status": quality_approval.get("status"),
            "catalog_approval_status": catalog_approval.get("status"),
            "strategy_file": strategy.get("path"),
            "strategy_sha256": strategy.get("sha256"),
            "strategy_params": json_text(strategy.get("params")),
            "schedule": execution.get("schedule"),
            "decision_time": decision_time_text(execution),
            "from_date": execution.get("from_date"),
            "execution_timing": execution.get("timing"),
            "missing_decision_policy": execution.get("missing_decision_policy"),
            "commission_bps": execution.get("commission_bps"),
            "slippage_bps": execution.get("slippage_bps"),
            "slippage_ticks": execution.get("slippage_ticks"),
            "bid_ask_spread_bps": execution.get("bid_ask_spread_bps"),
            "bid_ask_spread_ticks": execution.get("bid_ask_spread_ticks"),
            "volume_participation_limit": execution.get("volume_participation_limit"),
            "partial_fill_policy": execution.get("partial_fill_policy"),
            "intrabar_ambiguity_policy": execution.get("intrabar_ambiguity_policy"),
            "sizing_mode": portfolio.get("sizing_mode"),
            "initial_capital": portfolio.get("initial_capital"),
            **metrics_columns(metrics),
        }
    )


def metrics_columns(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "event_count",
        "trade_count",
        "order_count",
        "partial_fill_count",
        "volume_rejected_count",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "net_pnl",
        "final_equity_cash",
        "total_cost_cash",
        "turnover_notional",
        "max_position_notional",
        "exposure",
        "long_exposure",
        "short_exposure",
    ]
    return {key: metrics.get(key) for key in keys}


def failed_rows_from_batch_manifest(
    manifest_path: Path,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    experiment_id = payload.get("experiment_id")
    experiment_path = payload.get("output_dir") or str(manifest_path.parent)
    manifest_sha256 = None
    for failed in payload.get("failed_runs", []):
        if not isinstance(failed, dict):
            continue
        if manifest_sha256 is None:
            manifest_sha256 = file_sha256(manifest_path)
        rows.append(
            ordered_registry_row(
                {
                    "status": "failed",
                    "parent_experiment_id": experiment_id,
                    "parent_experiment_path": experiment_path,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_sha256,
                    "dataset_path": failed.get("dataset_path"),
                    "strategy_file": failed.get("strategy_file"),
                    "strategy_params": failed.get("params"),
                    "error_type": failed.get("error_type"),
                    "error_message": failed.get("error_message"),
                }
            )
        )
    return rows


def ordered_registry_row(values: dict[str, Any]) -> dict[str, Any]:
    return {column: values.get(column) for column in REGISTRY_COLUMNS}
