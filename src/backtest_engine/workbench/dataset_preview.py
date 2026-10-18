"""Builds bounded, chunked candle previews for the Workbench chart."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.workbench.datasets import get_workbench_dataset
from backtest_engine.workbench.preview_storage import (
    cached_preview_frame,
    cached_source_row_count,
    preview_timestamp,
)
from backtest_engine.workbench.preview_storage import downsample_ohlcv as downsample_ohlcv

DEFAULT_CHUNK_SIZE = 600
DEFAULT_MAX_POINTS = 12_000
MAX_CHUNK_SIZE = 2_000
MAX_PREVIEW_POINTS = 50_000


def dataset_preview_payload(
    output_root: Path,
    dataset_id: str,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    """Return one display chunk from a catalogued OHLCV dataset."""
    item = get_workbench_dataset(output_root, dataset_id)
    dataset_path = Path(str(item["dataset_path"]))
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset file not found: {dataset_path}")

    offset = query_int(query, "offset", 0, minimum=0)
    limit = query_int(
        query,
        "limit",
        DEFAULT_CHUNK_SIZE,
        minimum=1,
        maximum=MAX_CHUNK_SIZE,
    )
    max_points = query_int(
        query,
        "max_points",
        DEFAULT_MAX_POINTS,
        minimum=100,
        maximum=MAX_PREVIEW_POINTS,
    )
    stat = dataset_path.stat()
    cache_key = (str(dataset_path.resolve()), stat.st_mtime_ns, stat.st_size)
    source_timeframe = optional_text(item.get("timeframe"))
    chart_timeframe = query_optional_text(query, "timeframe") or source_timeframe
    sma_period = query_optional_int(query, "sma_period", minimum=1, maximum=10_000)
    ema_period = query_optional_int(query, "ema_period", minimum=1, maximum=10_000)
    bollinger_period = query_optional_int(query, "bollinger_period", minimum=1, maximum=10_000)
    requested_stddev = query_optional_float(query, "bollinger_stddev")
    bollinger_stddev = 2.0 if requested_stddev is None else requested_stddev
    inflation_rate = query_optional_float(query, "inflation_rate")
    timestamp_type = pl.read_parquet_schema(dataset_path)["ts"]
    inflation_base_ts = preview_timestamp(
        query_optional_datetime(query, "inflation_base_ts"), timestamp_type
    )
    from_ts = preview_timestamp(query_optional_datetime(query, "from_ts"), timestamp_type)
    till_ts = preview_timestamp(query_optional_datetime(query, "till_ts"), timestamp_type)
    if from_ts is not None and till_ts is not None and from_ts > till_ts:
        raise ValueError("preview from_ts must not be later than till_ts")
    source_row_count = cached_source_row_count(*cache_key)
    frame, resolved_timeframe, range_source_row_count, stride = cached_preview_frame(
        *cache_key,
        source_timeframe,
        chart_timeframe,
        sma_period,
        ema_period,
        bollinger_period,
        bollinger_stddev,
        inflation_rate,
        inflation_base_ts,
        from_ts,
        till_ts,
        preview_warmup_minutes(
            source_timeframe,
            chart_timeframe,
            max(sma_period or 0, bollinger_period or 0, (ema_period or 0) * 5),
        ),
        max_points,
    )
    chunk = frame.slice(offset, limit)
    next_offset = offset + chunk.height
    return {
        "dataset_id": dataset_id,
        "rows": chunk.to_dicts(),
        "offset": offset,
        "next_offset": next_offset,
        "preview_row_count": frame.height,
        "source_row_count": source_row_count,
        "range_source_row_count": range_source_row_count,
        "sampling_stride": stride,
        "exact": stride == 1,
        "timeframe": resolved_timeframe,
        "source_timeframe": source_timeframe,
        "overlays": {
            "sma_period": sma_period,
            "ema_period": ema_period,
            "bollinger_period": bollinger_period,
            "bollinger_stddev": bollinger_stddev if bollinger_period else None,
            "inflation_rate": inflation_rate,
            "inflation_base_ts": inflation_base_ts,
        },
        "done": next_offset >= frame.height,
    }


def preview_warmup_minutes(
    source_timeframe: str | None,
    chart_timeframe: str | None,
    warmup_bars: int,
) -> int:
    if warmup_bars < 1:
        return 0
    timeframe = chart_timeframe or source_timeframe
    if timeframe == "1d":
        return warmup_bars * 24 * 60
    if timeframe is None:
        return 0
    from backtest_engine.core.timeframes import parse_timeframe_minutes

    return parse_timeframe_minutes(timeframe) * warmup_bars


def query_int(
    query: dict[str, list[str]],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw = query.get(name, [str(default)])[0]
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be >= {minimum}{upper}")
    return value


def query_optional_datetime(
    query: dict[str, list[str]],
    name: str,
) -> datetime | None:
    if name not in query:
        return None
    raw = query[name][0]
    try:
        return datetime.fromisoformat(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO timestamp") from exc


def query_optional_text(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    return optional_text(values[0]) if values else None


def optional_text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def query_optional_int(
    query: dict[str, list[str]],
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    if name not in query:
        return None
    return query_int(query, name, minimum, minimum=minimum, maximum=maximum)


def query_optional_float(query: dict[str, list[str]], name: str) -> float | None:
    if name not in query:
        return None
    try:
        value = float(query[name][0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value
