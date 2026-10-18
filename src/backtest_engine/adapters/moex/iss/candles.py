"""Downloads MOEX candles and assembles security or nearest-futures inputs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backtest_engine.adapters.build_events import DatasetBuildObserver
from backtest_engine.adapters.models import DatasetBuildRequest, DatasetBuildResult
from backtest_engine.adapters.moex.iss.candle_transport import (
    MOEX_NATIVE_INTERVALS as MOEX_NATIVE_INTERVALS,
)
from backtest_engine.adapters.moex.iss.candle_transport import (
    RawRowsCallback,
)
from backtest_engine.adapters.moex.iss.candle_transport import (
    download_network_candles as download_network_candles,
)
from backtest_engine.adapters.moex.iss.candle_transport import (
    live_tail_requested as live_tail_requested,
)
from backtest_engine.adapters.moex.iss.catalog import (
    list_moex_boards as list_moex_boards,
)
from backtest_engine.adapters.moex.iss.catalog import (
    list_moex_engines as list_moex_engines,
)
from backtest_engine.adapters.moex.iss.catalog import (
    list_moex_futures_roots as list_moex_futures_roots,
)
from backtest_engine.adapters.moex.iss.catalog import (
    list_moex_instrument_rows,
)
from backtest_engine.adapters.moex.iss.catalog import (
    list_moex_instruments as list_moex_instruments,
)
from backtest_engine.adapters.moex.iss.catalog import (
    list_moex_markets as list_moex_markets,
)
from backtest_engine.adapters.moex.iss.client import iss_table_rows as iss_table_rows
from backtest_engine.adapters.moex.iss.contract_catalog import (
    contracts_for_root as contracts_for_root,
)
from backtest_engine.adapters.moex.iss.contract_selection import chain_audit
from backtest_engine.adapters.moex.iss.nearest_download import (
    download_nearest_futures_candles,
)
from backtest_engine.adapters.moex.iss.security_download import download_security_candles
from backtest_engine.adapters.moex.iss.snapshot_catalog import snapshot_instrument_rows
from backtest_engine.adapters.moex.pipeline.metadata import (
    attach_metadata_to_approval,
    write_moex_dataset_metadata,
)
from backtest_engine.adapters.moex.pipeline.models import MoexDataPipelineConfig
from backtest_engine.adapters.moex.pipeline.paths import (
    aggregate_timeframes,
    dataset_output_root,
    default_continuous_symbol,
    default_dataset_id,
    source_timeframe,
)
from backtest_engine.adapters.moex.pipeline.raw_storage import write_raw_snapshot
from backtest_engine.adapters.moex.pipeline.runner import run_moex_data_pipeline
from backtest_engine.adapters.moex.preview import MoexPreviewEmitter
from backtest_engine.data.store import parquet_metadata_index


def build_moex_dataset(
    request: DatasetBuildRequest,
    observer: DatasetBuildObserver | None = None,
) -> DatasetBuildResult:
    from backtest_engine.adapters.moex.adapter import MoexIssAdapter
    from backtest_engine.adapters.request_defaults import resolve_build_request

    request = resolve_build_request(request, MoexIssAdapter())
    preview = MoexPreviewEmitter(request, observer)
    if observer is not None:
        observer.progress(
            stage="dataset_download",
            message=f"downloading {request.security} market data",
            pct=0.15,
        )
    raw_path = write_candles_snapshot(
        request,
        on_rows=preview.consume,
        observer=observer,
    )
    preview.finish()
    dataset_id = request.dataset_id or default_dataset_id(request)
    continuous_symbol = request.continuous_symbol or default_continuous_symbol(request)
    base_timeframe = source_timeframe(request.timeframe)
    if observer is not None:
        observer.progress(
            stage="dataset_pipeline",
            message="normalizing bars and building nearest futures",
            pct=0.7,
        )
    result = run_moex_data_pipeline(
        MoexDataPipelineConfig(
            output_root=dataset_output_root(request),
            raw_snapshot_path=raw_path,
            dataset_id=dataset_id,
            continuous_symbol=continuous_symbol,
            base_timeframe=base_timeframe,
            aggregate_timeframes=aggregate_timeframes(request.timeframe, base_timeframe),
            gap_policy=request.gap_policy,
            gap_repair_policy=request.gap_repair_policy,
            source_delay_minutes=request.source_delay_minutes,
        )
    )
    row_count = int(parquet_metadata_index(result.nearest_path)["num_rows"])
    if observer is not None:
        observer.progress(
            stage="dataset_quality",
            message="auditing gaps, rolls, finality, and metadata",
            pct=0.85,
        )
    metadata = write_moex_dataset_metadata(
        output_root=result.output_root,
        dataset_path=result.nearest_path,
        instrument_rows=instrument_metadata_rows(request),
        expected_calendar_path=request.calendar_path,
        from_date=request.from_date,
        till_date=request.till_date,
    )
    quality = attach_metadata_to_approval(
        result.approval_path,
        result.manifest_path,
        metadata,
    )
    return DatasetBuildResult(
        adapter_id="moex",
        dataset_id=result.dataset_id,
        dataset_path=result.nearest_path,
        output_root=result.output_root,
        approval_status=str(quality["approval_status"]),
        approval_path=result.approval_path,
        manifest_path=result.manifest_path,
        raw_snapshot_path=result.raw_snapshot_path,
        timeframe=request.timeframe,
        row_count=row_count,
        quality=quality,
        instrument_registry_path=metadata.instrument_registry_path,
        instrument_registry_audit_path=metadata.instrument_registry_audit_path,
        calendar_path=metadata.calendar_path,
        calendar_audit_path=metadata.calendar_audit_path,
        source_as_of=result.source_as_of,
        finalized_through=result.finalized_through,
        bar_time_semantics="open_with_available_at",
    )


def write_candles_snapshot(
    request: DatasetBuildRequest,
    *,
    on_rows: RawRowsCallback | None = None,
    observer: DatasetBuildObserver | None = None,
) -> Path:
    rows = download_moex_candles(request, on_rows=on_rows, observer=observer)
    payload = {
        "source": "moex_iss",
        "generated_at": datetime.now(UTC).isoformat(),
        "request": request.model_dump(mode="json"),
        "chain_audit": snapshot_chain_audit(request, rows),
        "rows": rows,
    }
    return write_raw_snapshot(dataset_output_root(request) / "raw", payload)


def snapshot_chain_audit(
    request: DatasetBuildRequest,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if request.instrument_mode != "nearest_futures":
        return {}
    return chain_audit(
        contracts_for_root(request),
        roll_rule=request.roll_rule,
        selected_rows=rows,
    )


def download_moex_candles(
    request: DatasetBuildRequest,
    *,
    on_rows: RawRowsCallback | None = None,
    observer: DatasetBuildObserver | None = None,
) -> list[dict[str, Any]]:
    if request.instrument_mode == "nearest_futures":
        return download_nearest_futures_candles(
            request,
            load_security=lambda child, callback: download_moex_candles(
                child,
                on_rows=callback,
            ),
            on_rows=on_rows,
            observer=observer,
        )
    return download_security_candles(request, on_rows=on_rows)


def instrument_metadata_rows(request: DatasetBuildRequest) -> list[dict[str, Any]]:
    try:
        if request.instrument_mode == "nearest_futures":
            return contracts_for_root(request)
        rows = (
            snapshot_instrument_rows(
                engine=request.engine,
                market=request.market,
                board=request.board,
            )
            if request.cache_policy not in {"network", "refresh"}
            else []
        )
        if not rows and request.cache_policy in {"network", "refresh"}:
            rows = list_moex_instrument_rows(
                engine=request.engine,
                market=request.market,
                board=request.board,
                cache_policy=request.cache_policy,
                cache_dir=request.cache_dir,
            )
    except (OSError, ValueError):
        return [{"SECID": request.security}]
    selected = [row for row in rows if str(row.get("SECID")) == request.security]
    return selected or [{"SECID": request.security}]
