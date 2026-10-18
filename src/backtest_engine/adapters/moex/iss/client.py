"""Provides paginated, cached HTTP access to MOEX ISS tables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from moex_iss_adapter.normalizers import parse_moex_table

from backtest_engine.adapters.moex.iss.cache import DEFAULT_CACHE_DIR, fetch_iss_json_cached

ISS_BASE_URL = "https://iss.moex.com/iss"


def fetch_iss_json(
    url: str,
    params: dict[str, str] | None = None,
    *,
    cache_policy: str | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    return fetch_iss_json_cached(
        url,
        params,
        cache_policy=cache_policy or os.environ.get("MOEX_ISS_CACHE_POLICY", "prefer"),
        cache_dir=cache_dir or DEFAULT_CACHE_DIR,
    )


def iss_table_rows(payload: dict[str, Any], table_name: str) -> list[dict[str, Any]]:
    if table_name not in payload:
        return []
    return parse_moex_table(payload, table_name, normalize_column_names=False)


def cursor_total(payload: dict[str, Any], table_name: str) -> int | None:
    rows = iss_table_rows(payload, table_name)
    if not rows:
        return None
    raw_total = rows[0].get("TOTAL")
    return None if raw_total is None else int(raw_total)
