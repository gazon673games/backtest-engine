"""Persist exact Binance responses so interrupted historical builds can resume."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from backtest_engine.adapters.binance.transport import BinanceTransport
from backtest_engine.adapters.raw_json import write_gzip_json
from backtest_engine.core.exceptions import DatasetValidationError


class CachedTransport:
    def __init__(self, transport: BinanceTransport, root: Path, policy: str) -> None:
        if policy not in {"prefer", "network", "only"}:
            raise DatasetValidationError(f"Unsupported Binance cache policy: {policy}")
        self.transport = transport
        self.root = root
        self.policy = policy
        self.records: dict[str, dict[str, Any]] = {}

    def get(self, product: str, path: str, params: dict[str, str | int] | None = None) -> Any:
        request = {"product": product, "path": path, "params": params or {}}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        index = self.root / "requests" / f"{key}.json"
        historical = int((params or {}).get("endTime", 2**63)) < (
            int(datetime.now(UTC).timestamp() * 1000) - 86_400_000
        )
        if index.is_file() and (self.policy == "only" or self.policy == "prefer" and historical):
            record = json.loads(index.read_text(encoding="utf-8"))
            raw_path = self.root / record["file"]
            raw = raw_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise DatasetValidationError(f"Binance cached response hash mismatch: {raw_path}")
            self.records[key] = record
            return json.loads(gzip.decompress(raw))["response"]
        if self.policy == "only":
            raise DatasetValidationError(
                f"Binance offline cache missing: {product} {path} {params}"
            )
        response = self.transport.get(product, path, params)
        payload = {"request": request, "response": response}
        temporary = self.root / "objects" / f"{uuid4().hex}.pending.json.gz"
        write_gzip_json(temporary, payload)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        target = temporary.with_name(f"{digest}.json.gz")
        if target.exists():
            temporary.unlink()
        else:
            temporary.replace(target)
        record = {"request": request, "file": str(target.relative_to(self.root)), "sha256": digest}
        index.parent.mkdir(parents=True, exist_ok=True)
        partial = index.with_suffix(f".{uuid4().hex}.pending")
        partial.write_text(json.dumps(record), encoding="utf-8")
        partial.replace(index)
        self.records[key] = record
        return response

    def provenance(self) -> list[dict[str, Any]]:
        return [
            {**record, "path": str((self.root / record["file"]).resolve())}
            for record in self.records.values()
        ]
