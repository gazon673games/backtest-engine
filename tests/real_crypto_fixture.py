"""Replay the exact, hash-checked Binance responses captured for a real trading day."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from binance_fixture import exchange_info

RAW_ROOT = Path(__file__).parent / "fixtures/binance/solusdt_2024_04_01"
TARGETS = {"0": "0.25", "8": "0.5", "16": "0.75", "22": "0"}


class RecordedSolTransport:
    def __init__(self, raw_root: Path = RAW_ROOT) -> None:
        self.manifest = json.loads((raw_root / "manifest.json").read_text(encoding="utf-8"))
        self.interval = self.manifest.get("interval", "1h")
        self.responses = {}
        for record in self.manifest["responses"]:
            name = record.get("series") or record["name"]
            filename = record.get("file") or (name + ".json")
            body = (raw_root / filename).read_bytes()
            if filename.endswith(".gz"):
                assert hashlib.sha256(body).hexdigest() == record["compressed_sha256"], filename
                body = gzip.decompress(body)
            assert hashlib.sha256(body).hexdigest() == record["sha256"], filename
            rows = json.loads(body)
            assert len(rows) == record["count"], filename
            self.responses.setdefault(name, []).extend(rows)

    def get(self, product: str, path: str, params: dict[str, str | int] | None = None) -> Any:
        if path.endswith("exchangeInfo"):
            # Currency identity/filter metadata is a declared test policy, not
            # a claim to have captured the historical exchange specification.
            metadata = exchange_info(product)
            metadata["symbols"][0].update(symbol="SOLUSDT", baseAsset="SOL")
            return metadata
        assert params is not None and params["symbol"] == "SOLUSDT"
        if path.endswith("fundingRate"):
            assert product == "perpetual"
            rows = self.responses["perpetual_funding"]
            return [
                row
                for row in rows
                if params["startTime"] <= row["fundingTime"] <= params["endTime"]
            ][: params["limit"]]
        assert params["interval"] == self.interval
        if path.endswith("markPriceKlines"):
            assert product == "perpetual"
            key = "perpetual_marks"
        else:
            assert path.endswith("klines"), path
            key = product + "_klines"
        return [
            row for row in self.responses[key] if params["startTime"] <= row[0] <= params["endTime"]
        ][: params["limit"]]
