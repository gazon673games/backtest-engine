"""Build large histories one calendar month at a time into a single Parquet file."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from backtest_engine.adapters.binance.archive import archive_klines
from backtest_engine.adapters.binance.cache import CachedTransport
from backtest_engine.adapters.binance.download import (
    download_funding,
    download_klines,
    download_mark_klines,
    request_bounds,
)
from backtest_engine.adapters.binance.funding_prefix import recover_prefix_funding
from backtest_engine.adapters.binance.funding_recovery import recover_funding_prices
from backtest_engine.adapters.binance.mark_recovery import recover_missing_marks
from backtest_engine.adapters.binance.normalize import normalize_klines
from backtest_engine.adapters.binance.reuse import finalized_prefix
from backtest_engine.adapters.build_events import DatasetBuildObserver
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.timeframes import parse_timeframe_minutes
from backtest_engine.intraday.validation import validate_intraday_dataset


def month_windows(start_ms: int, end_ms: int):
    cursor = start_ms
    while cursor <= end_ms:
        stamp = datetime.fromtimestamp(cursor / 1000, UTC)
        next_month = datetime(stamp.year + (stamp.month == 12), stamp.month % 12 + 1, 1, tzinfo=UTC)
        boundary = int(next_month.timestamp() * 1000)
        yield stamp.strftime("%Y-%m"), cursor, min(end_ms, boundary - 1), boundary
        cursor = boundary


def build_bar_file(
    request: DatasetBuildRequest,
    transport: CachedTransport,
    metadata: dict[str, Any],
    path: Path,
    observer: DatasetBuildObserver | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start_ms, end_ms = request_bounds(request.from_date, request.till_date)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    end_ms = min(end_ms, now_ms)
    writer = None
    validation: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    previous_ts = None
    gap_count = missing_bars = mark_missing = mark_invalid = 0
    unpriced_funding = estimated_funding = 0
    temporary = path.with_suffix(".pending.parquet")
    try:
        for label, last, frame in bar_chunks(
            request, transport, metadata, path, sources, start_ms, end_ms, now_ms, observer
        ):
            chunk = validate_intraday_dataset(frame, path)
            step = parse_timeframe_minutes(request.timeframe) * 60
            timestamps = frame.get_column("ts").to_list()
            for current in timestamps:
                if previous_ts is not None:
                    delta = (current - previous_ts).total_seconds()
                    if delta <= 0:
                        raise DatasetValidationError(f"Binance history overlaps at {current}")
                    if delta > step:
                        gap_count += 1
                        missing_bars += max(1, int(delta / step) - 1)
                previous_ts = current
            if request.engine == "perpetual":
                unpriced_funding += frame.filter(
                    pl.col("funding_event") & pl.col("funding_price_rate").is_null()
                ).height
                estimated_funding += frame.filter(pl.col("funding_price_estimated")).height
                mark_columns = ("mark_open", "mark_high", "mark_low", "mark_price")
                frame = frame.with_columns(pl.col(*mark_columns).cast(pl.Float64))
                mark_missing += frame.filter(
                    pl.any_horizontal([pl.col(name).is_null() for name in mark_columns])
                ).height
                mark_invalid += frame.filter(
                    pl.any_horizontal(
                        [(~pl.col(name).is_finite()) | (pl.col(name) <= 0) for name in mark_columns]
                    )
                    | (pl.col("mark_low") > pl.min_horizontal("mark_open", "mark_price"))
                    | (pl.col("mark_high") < pl.max_horizontal("mark_open", "mark_price"))
                ).height
            if writer is None:
                validation = {**chunk, "row_count": 0}
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(temporary, frame.to_arrow().schema, compression="zstd")
            # Catalog observations can store the same columns in a different order.
            table = frame.select(writer.schema.names).to_arrow().cast(writer.schema)
            writer.write_table(table)
            validation["row_count"] += frame.height
            validation["max_ts"] = chunk["max_ts"]
            validation["finalized_through"] = frame.get_column("available_at").max().isoformat()
            if observer is not None:
                observer.progress(
                    stage="dataset_pipeline",
                    message=f"Binance {request.security}: {label}",
                    pct=0.1 + 0.55 * (last - start_ms) / max(1, end_ms - start_ms),
                )
        if writer is None:
            raise DatasetValidationError(
                f"Binance returned no finalized klines for {request.security}"
            )
        writer.close()
        writer = None
        temporary.replace(path)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    validation.update(
        {
            "gap_count": gap_count,
            "missing_bars": missing_bars,
            "missing_mark_bars": mark_missing,
            "invalid_mark_bars": mark_invalid,
            "unpriced_funding_bars": unpriced_funding,
            "estimated_funding_bars": estimated_funding,
        }
    )
    return validation, sources


def bar_chunks(
    request: DatasetBuildRequest,
    transport: CachedTransport,
    metadata: dict[str, Any],
    path: Path,
    sources: list[dict[str, Any]],
    start_ms: int,
    end_ms: int,
    now_ms: int,
    observer: DatasetBuildObserver | None,
) -> Iterator[tuple[str, int, pl.DataFrame]]:
    prefix = finalized_prefix(request, path.parent.parent, metadata, end_ms)
    first_ms = start_ms
    if prefix is not None:
        sources.append(prefix.provenance)
        first_ms = prefix.next_ms
        if observer is not None:
            observer.progress(
                stage="dataset_reuse", message="reading verified local history", pct=0.1
            )
        for frame in prefix.frames():
            if request.engine == "perpetual":
                if "funding_price_estimated" not in frame.columns:
                    frame = frame.with_columns(pl.lit(False).alias("funding_price_estimated"))
                if request.funding_price_policy == "minute_mark_open_estimate":
                    frame = recover_prefix_funding(frame, transport, request.security, sources)
                elif frame["funding_price_estimated"].any():
                    raise DatasetValidationError(
                        "Binance event_only cannot reuse estimated funding; "
                        "select cache_policy=network to rebuild from raw observations"
                    )
            yield "verified local history", int(frame["available_at"][-1].timestamp() * 1000), frame
    for month, first, last, boundary in month_windows(first_ms, end_ms):
        progress = (
            0.1 + 0.55 * (first - start_ms) / max(1, end_ms - start_ms),
            0.1 + 0.45 * (last - start_ms) / max(1, end_ms - start_ms),
        )
        if observer is not None:
            observer.progress(
                stage="dataset_download",
                message=f"fetching Binance {request.security}: {month}",
                pct=progress[0],
            )
        bars = history_rows(
            request,
            transport,
            sources,
            month,
            first,
            last,
            boundary,
            now_ms,
            observer=observer,
            progress_range=progress,
        )
        if not bars:
            continue
        funding, marks = [], []
        if request.engine == "perpetual":
            funding = download_funding(
                transport, symbol=request.security, start_ms=first, end_ms=last
            )
            marks = history_rows(
                request,
                transport,
                sources,
                month,
                first,
                last,
                boundary,
                now_ms,
                resource="markPriceKlines",
            )
            if request.archive_fallback_policy == "invalid_timestamps_and_missing_marks":
                marks = recover_missing_marks(request, transport, bars, marks, sources)
            if request.funding_price_policy == "minute_mark_open_estimate":
                funding = recover_funding_prices(
                    transport, symbol=request.security, events=funding, marks=marks, sources=sources
                )
        frame = normalize_klines(
            product=request.engine,
            symbol=request.security,
            timeframe=request.timeframe,
            metadata=metadata,
            klines=bars,
            funding=funding,
            mark_klines=marks,
        )
        yield month, last, frame


def history_rows(
    request: DatasetBuildRequest,
    transport: CachedTransport,
    sources: list[dict[str, Any]],
    month: str,
    first: int,
    last: int,
    boundary: int,
    now_ms: int,
    *,
    resource: str = "klines",
    observer: DatasetBuildObserver | None = None,
    progress_range: tuple[float, float] = (0.1, 0.6),
) -> list[list[Any]]:
    archive = None
    if request.history_source != "rest" and boundary <= now_ms:
        archive = archive_klines(
            transport.transport,
            product=request.engine,
            symbol=request.security,
            timeframe=request.timeframe,
            month=month,
            cache_root=transport.root,
            policy=request.cache_policy,
            resource=resource,
        )
    if archive is not None and request.archive_fallback_policy in {
        "invalid_timestamps",
        "invalid_timestamps_and_missing_marks",
    }:
        invalid = [row for row in archive[0] if int(row[6]) < int(row[0])]
        if invalid:
            if request.history_source != "auto":
                raise DatasetValidationError(
                    "Invalid Binance archive timestamps require auto REST fallback"
                )
            sources.append(
                {
                    **archive[1],
                    "status": "excluded_invalid_timestamps",
                    "invalid_rows": invalid,
                    "replacement_policy": "download_entire_month_from_rest",
                }
            )
            archive = None
    if archive is not None:
        rows, source = archive
        sources.append(source)
    else:
        if request.history_source == "archive" and boundary <= now_ms:
            raise DatasetValidationError(
                f"Binance {resource} archive unavailable: {request.security} {month}"
            )
        if resource == "markPriceKlines":
            rows = download_mark_klines(
                transport,
                symbol=request.security,
                timeframe=request.timeframe,
                start_ms=first,
                end_ms=last,
            )
        else:
            rows = download_klines(
                transport,
                product=request.engine,
                symbol=request.security,
                timeframe=request.timeframe,
                start_ms=first,
                end_ms=last,
                observer=observer,
                progress_range=progress_range,
            )
    return [row for row in rows if first <= int(row[0]) <= last and int(row[6]) <= now_ms]
