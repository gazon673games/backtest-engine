"""Read checksum-verified monthly OHLCV archives without extracting ZIP paths."""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from backtest_engine.adapters.binance.download import request_timeframe
from backtest_engine.core.exceptions import DatasetValidationError

ARCHIVE_ROOT = "https://data.binance.vision/data"


def archive_klines(
    transport: Any,
    *,
    product: str,
    symbol: str,
    timeframe: str,
    month: str,
    cache_root: Path,
    policy: str,
    resource: str = "klines",
) -> tuple[list[list[Any]], dict[str, Any]] | None:
    interval = request_timeframe(timeframe)
    kind = "spot" if product == "spot" else "futures/um"
    name = f"{symbol}-{interval}-{month}.zip"
    url = f"{ARCHIVE_ROOT}/{kind}/monthly/{resource}/{symbol}/{interval}/{name}"
    directory = cache_root / "archives" / product / resource / symbol / interval
    path = directory / name
    checksum_path = path.with_suffix(".zip.CHECKSUM")
    if policy != "network" and path.is_file() and checksum_path.is_file():
        raw, checksum = path.read_bytes(), checksum_path.read_bytes()
    else:
        if policy == "only":
            return None
        read_bytes = getattr(transport, "read_bytes", None)
        if read_bytes is None:
            return None
        raw = read_bytes(url, missing_ok=True)
        if raw is None:
            return None
        checksum = read_bytes(url + ".CHECKSUM", missing_ok=False)
        verify_checksum(raw, checksum, url)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid4().hex}.pending")
        temporary.write_bytes(raw)
        temporary.replace(path)
        checksum_path.write_bytes(checksum)
    digest = verify_checksum(raw, checksum, url)
    immutable = cache_root / "archive_objects" / f"{digest}.zip"
    immutable.parent.mkdir(parents=True, exist_ok=True)
    if not immutable.exists():
        immutable.write_bytes(raw)
    immutable_checksum = immutable.with_suffix(".zip.CHECKSUM")
    if not immutable_checksum.exists():
        immutable_checksum.write_bytes(checksum)
    rows = read_zip_klines(raw, name)
    return rows, {
        "source": "binance_public_archive",
        "url": url,
        "path": str(immutable.resolve()),
        "sha256": digest,
        "checksum_path": str(immutable_checksum.resolve()),
    }


def verify_checksum(raw: bytes, checksum: bytes, label: str) -> str:
    fields = checksum.decode("ascii").split()
    digest = hashlib.sha256(raw).hexdigest()
    if not fields or fields[0].lower() != digest:
        raise DatasetValidationError(f"Binance archive checksum mismatch: {label}")
    return digest


def read_zip_klines(raw: bytes, label: str) -> list[list[Any]]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = [item for item in archive.infolist() if item.filename.endswith(".csv")]
            if len(entries) != 1:
                raise DatasetValidationError(f"Binance archive requires one CSV: {label}")
            with archive.open(entries[0]) as source:
                rows = csv.reader(io.TextIOWrapper(source, encoding="utf-8-sig"))
                result = []
                for row in rows:
                    if row and row[0] in {"open_time", "Open time"}:
                        continue
                    if len(row) != 12:
                        raise DatasetValidationError(f"Invalid Binance kline row in {label}")
                    # Spot archive timestamps changed to microseconds in January 2025.
                    for index in (0, 6):
                        stamp = int(row[index])
                        row[index] = stamp // 1000 if stamp >= 10**15 else stamp
                    result.append(row)
                return result
    except (zipfile.BadZipFile, ValueError, UnicodeError) as exc:
        raise DatasetValidationError(f"Invalid Binance archive: {label}") from exc
