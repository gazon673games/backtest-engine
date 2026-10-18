"""Compact catalogs and validated native requests for both HTTP and MCP clients."""

from pathlib import Path
from typing import Any

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.workbench.agent_models import RunSpec
from backtest_engine.workbench.datasets import get_workbench_dataset, list_workbench_datasets
from backtest_engine.workbench.datasets import read_json as read_dataset_json
from backtest_engine.workbench.models import WorkbenchRunRequest
from backtest_engine.workbench.sdk_catalog import (
    CATALOG_PREFIX,
    get_catalog_record,
    list_catalog_records,
)
from backtest_engine.workbench.strategies import (
    get_saved_strategy,
    list_saved_strategies,
    metadata_from_source,
    strategies_root,
)


def page(items: list[dict], offset: int, limit: int) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError("offset must be >= 0; limit must be 1..100")
    return {
        "items": items[offset : offset + limit],
        "total": len(items),
        "offset": offset,
        "limit": limit,
        "next_offset": offset + limit if offset + limit < len(items) else None,
    }


def datasets(output_root: Path, query="", approved_only=True, offset=0, limit=20):
    items = list_workbench_datasets(output_root)["items"]
    selected = [
        item
        for item in items
        if query.casefold() in item["dataset_id"].casefold()
        and (not approved_only or item["approved_for_research"])
    ]
    fields = (
        "dataset_id",
        "dataset_type",
        "exchange",
        "engine",
        "market",
        "timeframe",
        "start_ts",
        "end_ts",
        "row_count",
        "quality_status",
        "approved_for_research",
        "exists",
    )
    return page([{k: item.get(k) for k in fields} for item in selected], offset, limit)


def strategies(output_root: Path, query="", offset=0, limit=20):
    items = list_saved_strategies(strategies_root(output_root))
    error = None
    try:
        items.extend(list_catalog_records())
    except (ValueError, OSError, SyntaxError, StrategyPluginError) as exc:
        error = str(exc)
    selected = [
        {k: item.get(k) for k in ("id", "name", "language", "params", "tags", "version")}
        for item in items
        if query.casefold() in str(item.get("name", "")).casefold()
    ]
    return {**page(selected, offset, limit), "catalog_error": error}


def strategy_record(output_root: Path, strategy_id: str):
    return (
        get_catalog_record(strategy_id)
        if strategy_id.startswith(CATALOG_PREFIX)
        else get_saved_strategy(strategies_root(output_root), strategy_id)
    )


def strategy_contract(output_root: Path, strategy_id: str, params: dict | None = None):
    record = strategy_record(output_root, strategy_id)
    metadata = metadata_from_source(
        record["source"],
        strategies_root(output_root),
        language=record.get("language", "javascript"),
        params={**record.get("params", {}), **(params or {})},
    )
    return {"strategy_id": strategy_id, "defaults": record.get("params", {}), "metadata": metadata}


def build_request(output_root: Path, spec: RunSpec) -> WorkbenchRunRequest:
    dataset = get_workbench_dataset(output_root, spec.dataset_id)
    if not dataset["approved_for_research"] or dataset["quality_status"] != "approved":
        path = dataset.get("quality_approval_path")
        approval = read_dataset_json(Path(path)) if path else {}
        reasons = "; ".join(str(value) for value in approval.get("violations", [])[:6])
        detail = f": {reasons}" if reasons else f" (status={dataset['quality_status']})"
        raise ValueError(f"dataset {spec.dataset_id} is not approved for research{detail}")
    record = strategy_record(output_root, spec.strategy_id)
    unknown = set(spec.execution_overrides) - set(WorkbenchRunRequest.model_fields)
    pinned = {
        "dataset_path",
        "dataset_id",
        "catalog_path",
        "catalog_enforcement",
        "quality_approval_path",
        "quality_approval_required",
        "script_path",
        "script_source",
        "script_language",
        "run_id",
        "output_root",
        "params",
        "initial_capital",
        "base_currency",
        "from_date",
    }
    if unknown or pinned & spec.execution_overrides.keys():
        invalid = sorted(unknown | (pinned & spec.execution_overrides.keys()))
        raise ValueError(f"invalid execution overrides: {invalid}")
    request = WorkbenchRunRequest.model_validate(
        {
            "dataset_path": dataset["dataset_path"],
            "dataset_id": spec.dataset_id,
            "catalog_path": dataset["catalog_path"],
            "catalog_enforcement": "strict",
            "quality_approval_path": dataset["quality_approval_path"],
            "quality_approval_required": True,
            "instrument_registry_path": dataset.get("instrument_registry_path"),
            "calendar_path": dataset.get("calendar_path"),
            "script_source": record["source"],
            "script_language": record.get("language", "javascript"),
            "params": {**record.get("params", {}), **spec.params},
            "from_date": str(spec.from_date),
            "initial_capital": spec.initial_capital,
            "base_currency": spec.base_currency,
            "output_root": output_root,
            **spec.execution_overrides,
        }
    )
    if not request.strict_runtime or request.allow_noncausal_same_close:
        raise ValueError("agent runs require causal execution and strict runtime")
    return request
