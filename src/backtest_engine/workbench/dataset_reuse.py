"""Freeze a compatible catalog prefix as an explicit queued ingestion input."""

from pathlib import Path

from backtest_engine.adapters.models import DatasetBuildRequest, DatasetPrefixInput
from backtest_engine.adapters.protocol import MarketDataAdapter
from backtest_engine.core.timeframes import parse_timeframe_minutes
from backtest_engine.data.store import file_sha256
from backtest_engine.workbench.datasets import list_workbench_datasets


def bind_reusable_dataset(
    output_root: Path, request: DatasetBuildRequest, adapter: MarketDataAdapter
) -> DatasetBuildRequest:
    if (
        not adapter.info().capabilities.supports_dataset_prefix_reuse
        or request.cache_policy == "network"
        or request.data_kind != "bars"
        or request.reuse_dataset is not None
    ):
        return request
    candidates = []
    for item in list_workbench_datasets(output_root)["items"]:
        metadata = item.get("metadata", {})
        if (
            metadata.get("adapter_id") != request.adapter_id
            or item.get("engine") != request.engine
            or (metadata.get("security") or item.get("continuous_symbol")) != request.security
            or item.get("quality_status") not in {"approved", "warning"}
            or metadata.get("bar_time_semantics") != "open_with_available_at"
            or not item.get("dataset_sha256")
            or (metadata.get("requested_from_date") or (item.get("start_ts") or "9999")[:10])
            > request.from_date
        ):
            continue
        if not item.get("timeframe") or parse_timeframe_minutes(item["timeframe"]) != (
            parse_timeframe_minutes(request.timeframe)
        ):
            continue
        candidates.append(item)
    if not candidates:
        return request
    item = max(candidates, key=lambda value: value.get("end_ts") or "")
    provenance = {
        key: item.get("metadata", {}).get(key)
        for key in ("manifest_path", "input_sha256", "historical_rules", "bar_policy")
    }
    if provenance["manifest_path"]:
        provenance["manifest_sha256"] = file_sha256(Path(provenance["manifest_path"]))
    prefix = DatasetPrefixInput(
        dataset_id=item["dataset_id"],
        dataset_path=item["dataset_path"],
        dataset_sha256=item["dataset_sha256"],
        adapter_id=request.adapter_id,
        engine=request.engine,
        symbol=request.security,
        timeframe=item["timeframe"],
        quality_status=item["quality_status"],
        provenance=provenance,
    )
    return request.model_copy(update={"reuse_dataset": prefix})
