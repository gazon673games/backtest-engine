"""Reads immutable candle snapshots produced by the sibling MOEX adapter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.cache import DEFAULT_CACHE_DIR
from backtest_engine.adapters.moex.iss.contract_selection import futures_root
from backtest_engine.adapters.moex.pipeline.paths import quote
from backtest_engine.adapters.moex.sessions import MOEX_TIMEZONE


@dataclass(frozen=True)
class CachedCandleSnapshot:
    path: Path
    filename_from: str
    filename_till: str
    data_from: str | None
    data_till: str | None
    rows: list[dict[str, Any]]
    confirmed_till: str | None = None


def cached_candle_rows(request: DatasetBuildRequest) -> list[dict[str, Any]] | None:
    snapshot = cached_candle_snapshot(request)
    return None if snapshot is None else snapshot.rows


def cached_candle_snapshot(request: DatasetBuildRequest) -> CachedCandleSnapshot | None:
    path = covering_snapshot(request)
    if path is None:
        return None
    return read_candle_snapshot(path, request)


def available_candle_snapshots(request: DatasetBuildRequest) -> list[CachedCandleSnapshot]:
    snapshots = []
    # New local observations override older raw rows at the same timestamp.
    for root in reversed(candle_roots(request)):
        directory = candle_directory(root, request)
        paths = directory.glob("from=*_till=*.json") if directory.exists() else []
        for path in sorted(paths, key=lambda value: value.stat().st_mtime_ns):
            bounds = filename_bounds(path)
            if bounds is not None and ranges_overlap(bounds, request):
                snapshot = read_candle_snapshot(path, request)
                if snapshot is not None:
                    snapshots.append(snapshot)
    return snapshots


def read_candle_snapshot(path: Path, request: DatasetBuildRequest) -> CachedCandleSnapshot | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bounds = filename_bounds(path)
    if bounds is None:
        return None
    rows = []
    timestamps = []
    for page in payload.get("pages", []):
        table = page.get("candles", {}) if isinstance(page, dict) else {}
        columns = table.get("columns", [])
        for values in table.get("data", []):
            row = dict(zip(columns, values, strict=False))
            timestamp = str(row.get("begin") or row.get("ts") or "")
            if timestamp:
                timestamps.append(timestamp)
            if request.from_date <= timestamp[:10] <= request.till_date:
                rows.append(row)
    return CachedCandleSnapshot(
        path=path,
        filename_from=bounds[0],
        filename_till=bounds[1],
        data_from=min(timestamps)[:10] if timestamps else None,
        data_till=max(timestamps)[:10] if timestamps else None,
        rows=rows,
        confirmed_till=payload.get("coverage", {}).get("till_date"),
    )


def covering_snapshot(request: DatasetBuildRequest) -> Path | None:
    candidates: list[tuple[int, int, int, str, str, Path]] = []
    for root in candle_roots(request):
        directory = candle_directory(root, request)
        for path in directory.glob("from=*_till=*.json") if directory.exists() else []:
            bounds = filename_bounds(path)
            if bounds is None or not ranges_overlap(bounds, request):
                continue
            covers = int(bounds[0] <= request.from_date and bounds[1] >= request.till_date)
            overlap = (
                date.fromisoformat(min(bounds[1], request.till_date))
                - date.fromisoformat(max(bounds[0], request.from_date))
            ).days
            local = int(root == local_candle_root(request) and bool(covers))
            candidates.append((local, covers, overlap, bounds[1], bounds[0], path))
    return max(candidates)[5] if candidates else None


def local_candle_root(request: DatasetBuildRequest) -> Path:
    return (request.cache_dir or DEFAULT_CACHE_DIR) / "candle_snapshots"


def candle_roots(request: DatasetBuildRequest) -> list[Path]:
    roots = [local_candle_root(request), raw_candle_root()]
    return [root for root in roots if root is not None and root.exists()]


def candle_directory(root: Path, request: DatasetBuildRequest) -> Path:
    return (
        root
        / f"engine={quote(str(request.engine))}"
        / f"market={quote(str(request.market))}"
        / f"board={quote(str(request.board))}"
        / f"symbol={quote(request.security)}"
        / f"timeframe={quote(request.timeframe)}"
    )


def save_candle_snapshot(request: DatasetBuildRequest, rows: list[dict[str, Any]]) -> None:
    directory = candle_directory(local_candle_root(request), request)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    today = now.astimezone(MOEX_TIMEZONE).date()
    first = date.fromisoformat(request.from_date).isoformat()
    till = min(date.fromisoformat(request.till_date).isoformat(), today.isoformat())
    path = directory / f"from={first}_till={till}.json"
    columns = sorted({key for row in rows for key in row})
    payload = {
        "source": "moex_iss_local_observations",
        "retrieved_at": now.isoformat(),
        "request": request.model_dump(mode="json"),
        # A successful query records coverage; it does not certify quality or fill gaps.
        "coverage": {
            "from_date": request.from_date,
            "till_date": min(till, (today - timedelta(days=1)).isoformat()),
        },
        "pages": [
            {
                "candles": {
                    "columns": columns,
                    "data": [[row.get(key) for key in columns] for row in rows],
                }
            }
        ],
    }
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot_contract_rows(
    request: DatasetBuildRequest,
    security_root: str,
) -> list[dict[str, Any]]:
    contracts = {}
    symbol_dirs = set()
    for root in candle_roots(request):
        board_root = (
            root
            / f"engine={request.engine}"
            / f"market={request.market}"
            / f"board={request.board}"
        )
        symbol_dirs.update(board_root.glob("symbol=*") if board_root.exists() else [])
    for symbol_dir in symbol_dirs:
        symbol = symbol_dir.name.removeprefix("symbol=")
        if futures_root(symbol) != security_root.upper():
            continue
        bounds = contract_snapshot_bounds(symbol_dir / f"timeframe={request.timeframe}", request)
        if bounds is None or bounds[1] < request.from_date:
            continue
        previous = contracts.get(symbol, {})
        contracts[symbol] = {
            "SECID": symbol,
            "snapshot_from": min(previous.get("snapshot_from", bounds[0]), bounds[0]),
            "snapshot_till": max(previous.get("snapshot_till", bounds[1]), bounds[1]),
            "registry_source": "candle_observation_only",
        }
    return sorted(contracts.values(), key=lambda row: str(row["SECID"]))


def contract_snapshot_bounds(
    directory: Path,
    request: DatasetBuildRequest,
) -> tuple[str, str] | None:
    candidates = []
    for path in directory.glob("from=*_till=*.json") if directory.exists() else []:
        bounds = filename_bounds(path)
        if bounds is not None and bounds[0] <= request.till_date:
            candidates.append(bounds)
    if not candidates:
        return None
    return min(start for start, _ in candidates), max(end for _, end in candidates)


def ranges_overlap(bounds: tuple[str, str], request: DatasetBuildRequest) -> bool:
    return bounds[0] <= request.till_date and bounds[1] >= request.from_date


def raw_candle_root() -> Path | None:
    configured = os.environ.get("WORKBENCH_MOEX_RAW_ROOT")
    candidates = [
        Path(configured) if configured else None,
        Path(__file__).resolve().parents[6]
        / "moex_iss_adapter"
        / "data"
        / "raw"
        / "moex"
        / "candles",
    ]
    return next((path for path in candidates if path is not None and path.exists()), None)


def filename_bounds(path: Path) -> tuple[str, str] | None:
    stem = path.stem
    if not stem.startswith("from=") or "_till=" not in stem:
        return None
    start, end = stem.removeprefix("from=").split("_till=", 1)
    return start, end
