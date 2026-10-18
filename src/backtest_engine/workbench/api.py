"""Translates HTTP payloads into exchange-neutral Workbench application calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.registry import get_adapter, list_adapters
from backtest_engine.adapters.request_defaults import catalog_choice, resolve_build_request
from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.workbench.compare import compare_workbench_runs
from backtest_engine.workbench.dataset_build_preview import read_dataset_build_preview
from backtest_engine.workbench.dataset_jobs import submit_dataset_build
from backtest_engine.workbench.dataset_preview import (
    dataset_preview_payload as dataset_preview_payload,
)
from backtest_engine.workbench.dataset_reuse import bind_reusable_dataset
from backtest_engine.workbench.datasets import (
    get_workbench_dataset,
    list_workbench_datasets,
    register_workbench_dataset,
)
from backtest_engine.workbench.jobs import cancel_job, list_jobs, submit_job
from backtest_engine.workbench.models import WorkbenchRunRequest
from backtest_engine.workbench.progress import read_progress
from backtest_engine.workbench.proxy_settings import (
    proxy_settings_payload,
    save_proxy_settings_payload,
)
from backtest_engine.workbench.research import research_report, submit_sweep
from backtest_engine.workbench.runner import generated_run_id, safe_run_id
from backtest_engine.workbench.sdk_catalog import (
    CATALOG_PREFIX,
    freeze_run_request,
    get_catalog_record,
    list_catalog_records,
)
from backtest_engine.workbench.strategies import (
    get_saved_strategy,
    list_saved_strategies,
    metadata_from_source,
    save_strategy,
    strategies_root,
)
from backtest_engine.workbench.validation import validate_workbench_run

settings_proxy_payload = proxy_settings_payload
settings_proxy_save_payload = save_proxy_settings_payload
dataset_list_payload = list_workbench_datasets
dataset_get_payload = get_workbench_dataset
async_dataset_build_payload = submit_dataset_build
jobs_payload = list_jobs
sweep_submit_payload = submit_sweep
research_report_payload = research_report
validation_payload = validate_workbench_run


def adapters_payload() -> dict[str, Any]:
    return {"adapters": [item.model_dump() for item in list_adapters()]}


def strategy_list_payload(output_root: Path) -> dict[str, Any]:
    items = list_saved_strategies(strategies_root(output_root))
    try:
        return {"items": [*items, *list_catalog_records()]}
    except (StrategyPluginError, ValueError, OSError, SyntaxError) as error:
        return {"items": items, "catalog_error": str(error)}


def strategy_get_payload(output_root: Path, strategy_id: str) -> dict[str, Any]:
    from backtest_engine.workbench.sdk_editing import editor_record

    record = (
        get_catalog_record(strategy_id)
        if strategy_id.startswith(CATALOG_PREFIX)
        else get_saved_strategy(strategies_root(output_root), strategy_id)
    )
    return editor_record(record)


def strategy_save_payload(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": "completed", "strategy": save_strategy(strategies_root(output_root), payload)}


def strategy_metadata_payload(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metadata = metadata_from_source(
        str(payload.get("source", "")),
        strategies_root(output_root),
        language=payload.get("language", "javascript"),
        params=payload.get("params"),
    )
    return {"status": "completed", "metadata": metadata}


def progress_payload(output_root: Path, run_id: str) -> dict[str, Any]:
    return read_progress(output_root, safe_run_id(run_id))


def adapter_payload(
    adapter_id: str,
    path: str,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    adapter = get_adapter(adapter_id)
    search = first_query(query, "query", "")
    if path.endswith("/engines"):
        return {"items": [item.model_dump() for item in adapter.engines()]}
    engine = catalog_choice(
        first_query(query, "engine", "") or adapter.info().capabilities.default_engine,
        adapter.engines(),
        "engine",
    )
    if path.endswith("/history-range"):
        provider = getattr(adapter, "history_range", None)
        if not adapter.info().capabilities.supports_history_range or not callable(provider):
            raise ValueError(f"Adapter {adapter_id} does not provide historical availability")
        result = provider(
            engine=engine,
            symbol=first_query(query, "symbol", ""),
            timeframe=first_query(query, "timeframe", "5m"),
        )
        return result.model_dump(mode="json")
    if path.endswith("/markets"):
        return {"items": [item.model_dump() for item in adapter.markets(engine)]}
    market = catalog_choice(
        first_query(query, "market", "") or None, adapter.markets(engine), "market"
    )
    if path.endswith("/boards"):
        return {"items": [item.model_dump() for item in adapter.boards(engine, market)]}
    board = catalog_choice(
        first_query(query, "board", "") or None, adapter.boards(engine, market), "board"
    )
    if path.endswith("/instruments"):
        return {
            "items": adapter.instruments(engine=engine, market=market, board=board, query=search)
        }
    resource = path.rsplit("/", 1)[-1]
    if resource in adapter.info().capabilities.instrument_catalogs.values():
        catalog = getattr(adapter, resource.replace("-", "_"), None)
        if not callable(catalog):
            raise ValueError(f"Adapter {adapter_id} declares unavailable catalog: {resource}")
        return {"items": catalog(engine=engine, market=market, board=board, query=search)}
    raise ValueError(f"unknown adapter endpoint: {adapter_id}")


def dataset_build_payload(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("output_root", str(output_root.parent / "datasets"))
    request = DatasetBuildRequest.model_validate(payload)
    adapter = get_adapter(request.adapter_id)
    request = resolve_build_request(request, adapter)
    request = bind_reusable_dataset(output_root, request, adapter)
    result = adapter.build_dataset(request)
    result = register_workbench_dataset(output_root, request, result)
    return {"status": "completed", **result.model_dump(mode="json")}


def dataset_build_live_preview_payload(
    output_root: Path,
    job_id: str,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    safe_job_id = safe_run_id(job_id)
    raw_after = query.get("after", ["0"])[0]
    try:
        after = int(raw_after)
    except (TypeError, ValueError) as exc:
        raise ValueError("after must be an integer") from exc
    if after < 0:
        raise ValueError("after must be >= 0")
    return read_dataset_build_preview(output_root, safe_job_id, after=after)


def async_run_payload(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("output_root", str(output_root))
    request = freeze_run_request(WorkbenchRunRequest.model_validate(payload))
    payload = request.model_dump(mode="json")
    run_id = safe_run_id(request.run_id or generated_run_id(request))
    payload["run_id"] = run_id
    WorkbenchRunRequest.model_validate(payload)
    job = submit_job(output_root, run_id, "run", payload)
    return {"status": "accepted", "run_id": run_id, "job": job}


def cancel_job_payload(output_root: Path, job_id: str) -> dict[str, Any]:
    return cancel_job(output_root, safe_run_id(job_id))


def compare_payload(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    run_ids = [safe_run_id(str(value)) for value in payload.get("run_ids", [])]
    return compare_workbench_runs(output_root, run_ids)


parse_query = parse_qs


def first_query(query: dict[str, list[str]], key: str, default: str) -> str:
    values = query.get(key)
    return values[0] if values else default
