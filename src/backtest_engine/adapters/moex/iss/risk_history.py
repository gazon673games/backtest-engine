"""Download and pin the public NCC historical first-level market risk rates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.client import (
    ISS_BASE_URL,
    cursor_total,
    fetch_iss_json,
    iss_table_rows,
)
from backtest_engine.adapters.moex.pipeline.paths import quote


@dataclass(frozen=True)
class RiskHistory:
    asset: str
    rows: tuple[dict[str, Any], ...]
    sources: tuple[dict[str, Any], ...]


def download_risk_history(asset: str, request: DatasetBuildRequest, reports: Path) -> RiskHistory:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", asset):
        raise ValueError(f"Invalid MOEX risk asset code: {asset!r}")
    url = f"{ISS_BASE_URL}/rms/engines/futures/objects/limits/{quote(asset)}.json"
    # The previous known regime is needed when a rate changes during the first day.
    first = date.fromisoformat(request.from_date) - timedelta(days=32)
    params = {
        "from": first.isoformat(),
        "till": request.till_date,
        "iss.meta": "off",
        "limit": "1000",
    }
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    sources = []
    start = 0
    raw_root = reports / "risk_history" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    for _page in range(100):
        page_params = {**params, "start": str(start)}
        payload = fetch_iss_json(
            url, page_params, cache_policy=request.cache_policy, cache_dir=request.cache_dir
        )
        if "limits" not in payload:
            raise ValueError(f"MOEX risk history for {asset}: limits table is missing")
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        raw_path = raw_root / f"{asset}_{start}_{digest[:16]}.json"
        raw_path.write_text(
            json.dumps({"url": url, "params": page_params, "payload": payload}, ensure_ascii=False),
            encoding="utf-8",
        )
        page = iss_table_rows(payload, "limits")
        sources.append({"url": url, "params": page_params, "sha256": digest, "path": str(raw_path)})
        cursors = iss_table_rows(payload, "limits.cursor")
        if not cursors or int(cursors[0]["INDEX"]) != start:
            raise ValueError(f"Invalid MOEX risk history cursor for {asset} at {start}")
        for raw_row in page:
            row = validate_rate_row(raw_row, asset)
            if not first.isoformat() <= row["tradedate"] <= request.till_date:
                raise ValueError(f"MOEX risk history date is outside the request for {asset}")
            row["source_sha256"] = digest
            key = (row["tradedate"], row["updatetime"])
            previous = rows.get(key)
            if previous and any(previous[name] != row[name] for name in ("mr1", "lk1")):
                raise ValueError(f"Conflicting MOEX risk history for {asset}: {key}")
            rows[key] = row
        total = cursor_total(payload, "limits.cursor")
        if not page:
            if total is not None and start < total:
                raise ValueError(f"Truncated MOEX risk history for {asset} at row {start}/{total}")
            break
        start += len(page)
        if total is None:
            raise ValueError(f"MOEX risk history for {asset}: pagination cursor is missing")
        if start >= total:
            break
    else:
        raise ValueError(f"MOEX risk history for {asset} exceeds 100 pages")
    if not rows:
        raise ValueError(f"MOEX risk history for {asset} is empty for the requested period")
    return RiskHistory(asset, tuple(rows[key] for key in sorted(rows)), tuple(sources))


def validate_rate_row(row: dict[str, Any], asset: str) -> dict[str, Any]:
    from decimal import Decimal, InvalidOperation

    if str(row.get("assetcode", "")).upper() != asset.upper():
        raise ValueError(f"MOEX risk history asset mismatch: expected {asset}")
    try:
        date.fromisoformat(str(row["tradedate"]))
        timestamp = datetime.fromisoformat(str(row["updatetime"]))
        rates = [Decimal(str(row[name])) for name in ("mr1", "mr2", "mr3")]
        limits = [Decimal(str(row[name])) for name in ("lk1", "lk2")]
        valid = (
            timestamp.tzinfo is None
            and all(value.is_finite() for value in [*rates, *limits])
            and 0 < rates[0] <= rates[1] <= rates[2] <= 1
            and 0 < limits[0] <= limits[1]
        )
    except (KeyError, ValueError, InvalidOperation) as error:
        raise ValueError(f"Invalid MOEX risk history for {asset}: {row}") from error
    if not valid:
        raise ValueError(f"Invalid MOEX risk history rates/timestamps for {asset}: {row}")
    return dict(row)
