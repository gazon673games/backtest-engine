from __future__ import annotations

import re
import urllib.parse
from pathlib import Path
from typing import Any

from backtest_engine.adapters.models import DatasetBuildRequest

MOEX_NATIVE_TIMEFRAMES = frozenset({"1m", "10m", "60m", "1d"})


def source_timeframe(timeframe: str) -> str:
    return timeframe if timeframe in MOEX_NATIVE_TIMEFRAMES else "1m"


def aggregate_timeframes(timeframe: str, base_timeframe: str) -> list[str]:
    return [] if timeframe == base_timeframe else [timeframe]


def default_dataset_id(request: DatasetBuildRequest) -> str:
    parts = (
        "moex",
        request.market,
        request.security,
        request.instrument_mode,
        request.timeframe,
        request.from_date,
        request.till_date,
        request.roll_rule,
    )
    return ":".join(parts).lower()


def default_continuous_symbol(request: DatasetBuildRequest) -> str:
    mode = "NEAREST" if request.instrument_mode == "nearest_futures" else "SECURITY"
    return f"{request.security}_{mode}_{request.timeframe}".upper()


def dataset_output_root(request: DatasetBuildRequest) -> Path:
    return request.output_root / safe_path(default_dataset_id(request))


def safe_path(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "dataset"


def optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")
