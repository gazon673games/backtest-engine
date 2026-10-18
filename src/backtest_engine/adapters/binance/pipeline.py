"""Builds auditable spot or perpetual Binance datasets."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from backtest_engine.adapters.binance.cache import CachedTransport
from backtest_engine.adapters.binance.download import (
    canonical_timeframe,
    download_aggregate_trades,
    request_bounds,
)
from backtest_engine.adapters.binance.normalize import (
    instrument_payload,
    normalize_aggregate_trades,
    normalize_quote_frame,
    symbol_metadata,
)
from backtest_engine.adapters.binance.normalize_book import normalize_order_book_frame
from backtest_engine.adapters.binance.quality import quality_payload
from backtest_engine.adapters.binance.rules import build_rule_registry
from backtest_engine.adapters.binance.stream import build_bar_file
from backtest_engine.adapters.binance.transport import BinanceTransport, endpoint
from backtest_engine.adapters.build_events import DatasetBuildObserver
from backtest_engine.adapters.models import DatasetBuildRequest, DatasetBuildResult
from backtest_engine.adapters.raw_json import write_gzip_json
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.intraday.validation import validate_intraday_dataset


def build_binance_dataset(
    request: DatasetBuildRequest,
    transport: BinanceTransport,
    observer: DatasetBuildObserver | None = None,
) -> DatasetBuildResult:
    product = validate_request(request)
    symbol = request.security.strip().upper()
    request = request.model_copy(
        update={"security": symbol, "timeframe": canonical_timeframe(request.timeframe)}
    )
    storage_id = request.dataset_id or default_dataset_id(product, symbol, request.timeframe)
    build_id = uuid4().hex
    dataset_id = request.dataset_id or f"{storage_id}-{build_id[:12]}"
    cache = CachedTransport(
        transport, request.output_root / storage_id / "raw_cache", request.cache_policy
    )
    output_root = request.output_root / storage_id / f"build-{build_id}"
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / "market_data.parquet"
    retrieved_at = datetime.now(UTC)
    publish_progress(observer, "dataset_prepare", f"preparing Binance {symbol}", 0.1)
    exchange_info = cache.get(product, endpoint(product, "exchangeInfo"))
    metadata = symbol_metadata(exchange_info, symbol)
    metadata_spec = instrument_payload(product, metadata)
    registry_specs, rule_audit = build_rule_registry(
        product=product,
        symbol=symbol,
        metadata_spec=metadata_spec,
        rules_path=request.rules_path,
        enforcement=request.rules_enforcement,
        from_date=request.from_date,
        till_date=request.till_date,
        retrieved_at=retrieved_at,
    )
    start_ms, end_ms = request_bounds(request.from_date, request.till_date)
    sources: list[dict[str, Any]] = []
    frame = None
    trades: list[dict[str, Any]] = []
    if request.data_kind == "bars":
        validation, sources = build_bar_file(request, cache, metadata, dataset_path, observer)
    elif request.data_kind == "trades":
        trades = download_aggregate_trades(
            cache,
            product=product,
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        frame = normalize_aggregate_trades(
            product=product,
            symbol=symbol,
            metadata=metadata,
            trades=trades,
        )
    elif request.data_kind == "quotes":
        assert request.source_path is not None
        frame = normalize_quote_frame(
            product=product,
            symbol=symbol,
            metadata=metadata,
            frame=read_source_frame(request.source_path),
        )
    else:
        assert request.source_path is not None
        frame = normalize_order_book_frame(
            product=product,
            symbol=symbol,
            metadata=metadata,
            frame=read_source_frame(request.source_path),
        )
    publish_progress(observer, "dataset_pipeline", "normalizing finalized bars", 0.65)
    raw_path = output_root / "adapter_raw.json.gz"
    write_gzip_json(
        raw_path,
        {
            "source": "binance_rest",
            "retrieved_at": datetime.now(UTC).isoformat(),
            "request": request.model_dump(mode="json"),
            "exchange_symbol": metadata,
            "raw_responses": cache.provenance(),
            "archives": sources,
            "aggregate_trades": trades,
            "source_path": str(request.source_path) if request.source_path else None,
            "source_sha256": sha256(request.source_path) if request.source_path else None,
            "rules_path": str(request.rules_path) if request.rules_path else None,
            "rules_sha256": sha256(request.rules_path) if request.rules_path else None,
        },
        indent=2,
    )
    if frame is not None:
        frame.write_parquet(dataset_path)
        validation = validate_intraday_dataset(frame, dataset_path)
    registry_path = output_root / "instrument_registry.json"
    write_json(registry_path, {"instruments": registry_specs})
    registry_audit_path = output_root / "instrument_registry_audit.json"
    write_json(registry_audit_path, rule_audit)
    calendar_path = output_root / "session_calendar.json"
    write_json(
        calendar_path,
        {
            "source": "binance_24x7",
            "timezone": "UTC",
            "trading_calendar": "crypto_24x7",
            "trading_days_per_year": 365,
            "holiday_dates": [],
            "short_day_dates": [],
            "main_session_start": "00:00",
            "main_session_end": "23:59",
            "evening_session_start": None,
            "evening_session_end": None,
        },
    )
    quality = quality_payload(frame, request, validation, rule_audit)
    approval_path = output_root / "quality_approval.json"
    write_json(approval_path, quality)
    manifest_path = output_root / "dataset_manifest.json"
    write_json(
        manifest_path,
        manifest_payload(
            request=request,
            product=product,
            symbol=symbol,
            dataset_path=dataset_path,
            raw_path=raw_path,
            registry_path=registry_path,
            registry_audit_path=registry_audit_path,
            approval_path=approval_path,
            validation=validation,
            calendar_path=calendar_path,
        ),
    )
    (output_root / "adapter_raw.json").unlink(missing_ok=True)
    preview = frame if frame is not None else pl.read_parquet(dataset_path, n_rows=5000)
    publish_preview(observer, preview.head(5000), request.timeframe)
    publish_progress(observer, "dataset_quality", quality["approval_status"], 1.0)
    finalized = (
        validation.get("finalized_through") or preview.get_column("available_at").max().isoformat()
    )
    return DatasetBuildResult(
        adapter_id="binance",
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        output_root=output_root,
        approval_status=str(quality["approval_status"]),
        approval_path=approval_path,
        manifest_path=manifest_path,
        raw_snapshot_path=raw_path,
        timeframe=request.timeframe,
        row_count=int(validation["row_count"]),
        quality=quality,
        instrument_registry_path=registry_path,
        instrument_registry_audit_path=registry_audit_path,
        calendar_path=calendar_path,
        source_as_of=retrieved_at.isoformat(),
        finalized_through=finalized,
        bar_time_semantics="open_with_available_at",
    )


def validate_request(request: DatasetBuildRequest) -> str:
    if request.funding_price_policy != "event_only" and (
        request.engine != "perpetual" or request.data_kind != "bars"
    ):
        raise DatasetValidationError("Funding price recovery requires perpetual bars")
    if request.history_source not in {"auto", "rest", "archive"}:
        raise DatasetValidationError("Binance history_source must be auto, rest, or archive")
    if not request.security.strip().isalnum():
        raise DatasetValidationError("Binance security must be an alphanumeric trading symbol")
    if request.adapter_id != "binance":
        raise DatasetValidationError("Binance adapter requires adapter_id=binance")
    if request.engine not in {"spot", "perpetual"}:
        raise DatasetValidationError("Binance engine must be spot or perpetual")
    if request.data_kind not in {"bars", "trades", "quotes", "book"}:
        raise DatasetValidationError("Binance data_kind must be bars, trades, quotes, or book")
    if request.data_kind in {"quotes", "book"}:
        if request.source_path is None or not request.source_path.is_file():
            raise DatasetValidationError(
                f"Historical Binance {request.data_kind} requires an existing source_path archive"
            )
    if request.rules_path is not None and not request.rules_path.is_file():
        raise DatasetValidationError(f"Binance rules_path does not exist: {request.rules_path}")
    return request.engine


def manifest_payload(**values: Any) -> dict[str, Any]:
    request = values["request"]
    return {
        "manifest_version": 2,
        "adapter": "binance",
        "product": values["product"],
        "symbol": values["symbol"],
        "request": request.model_dump(mode="json"),
        "validation": values["validation"],
        "files": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in {
                "dataset": values["dataset_path"],
                "raw": values["raw_path"],
                "instrument_registry": values["registry_path"],
                "instrument_registry_audit": values["registry_audit_path"],
                "quality_approval": values["approval_path"],
                "calendar": values["calendar_path"],
            }.items()
        },
    }


def default_dataset_id(product: str, symbol: str, timeframe: str) -> str:
    return f"binance-{product}-{symbol.lower()}-{timeframe.lower()}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def read_source_frame(path: Path) -> pl.DataFrame:
    try:
        if path.suffix.lower() == ".csv":
            return pl.read_csv(path, try_parse_dates=True)
        if path.suffix.lower() == ".parquet":
            return pl.read_parquet(path)
    except Exception as exc:
        raise DatasetValidationError(f"Failed to read Binance archive: {path}") from exc
    raise DatasetValidationError("Binance archive must be CSV or Parquet")


def publish_progress(
    observer: DatasetBuildObserver | None,
    stage: str,
    message: str,
    pct: float,
) -> None:
    if observer is not None:
        observer.progress(stage=stage, message=message, pct=pct)


def publish_preview(
    observer: DatasetBuildObserver | None,
    frame: pl.DataFrame,
    timeframe: str,
) -> None:
    if observer is not None:
        observer.preview(
            frame.select("ts", "open", "high", "low", "close", "volume").to_dicts(),
            timeframe=timeframe,
        )
