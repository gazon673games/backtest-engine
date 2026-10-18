"""Loads the approved local MOEX FORTS instrument snapshot for discovery and builds."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

FALLBACK_CONTRACTS = (
    ("CRU6", "CNY", "2026-09-17", 1000.0, 0.001, 1.0),
    ("CRZ6", "CNY", "2026-12-17", 1000.0, 0.001, 1.0),
    ("CRH7", "CNY", "2027-03-18", 1000.0, 0.001, 1.0),
    ("CRM7", "CNY", "2027-06-17", 1000.0, 0.001, 1.0),
    ("SIU6", "Si", "2026-09-17", 1000.0, 1.0, 1.0),
    ("SIZ6", "Si", "2026-12-17", 1000.0, 1.0, 1.0),
    ("SIH7", "Si", "2027-03-18", 1000.0, 1.0, 1.0),
    ("SIM7", "Si", "2027-06-17", 1000.0, 1.0, 1.0),
)


def snapshot_instrument_rows(
    *,
    engine: str,
    market: str,
    board: str,
) -> list[dict[str, Any]]:
    if (engine.lower(), market.lower(), board.upper()) != ("futures", "forts", "RFUD"):
        return []
    records = load_snapshot_records()
    return [normalized_row(record) for record in records if record.get("symbol")]


def load_snapshot_records() -> list[dict[str, Any]]:
    path = snapshot_path()
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("instruments", []) if isinstance(payload, dict) else []
        if records:
            return [record for record in records if isinstance(record, dict)]
    return [
        {
            "symbol": symbol,
            "assetcode": asset,
            "last_trade_date": expiry,
            "lot_size": lot_size,
            "tick_size": tick_size,
            "tick_value": tick_value,
            "source": "bundled_moex_snapshot",
        }
        for symbol, asset, expiry, lot_size, tick_size, tick_value in FALLBACK_CONTRACTS
    ]


def snapshot_path() -> Path | None:
    configured = os.environ.get("WORKBENCH_MOEX_SPECS_PATH")
    candidates = [
        Path(configured) if configured else None,
        Path(__file__).resolve().parents[5]
        / "configs"
        / "generated"
        / "moex_forts_instrument_specs.json",
    ]
    return next((path for path in candidates if path is not None and path.exists()), None)


def normalized_row(record: dict[str, Any]) -> dict[str, Any]:
    symbol = str(record.get("symbol", ""))
    asset = str(record.get("assetcode", ""))
    expiry = record.get("last_trade_date") or record.get("expiration_date")
    return {
        "SECID": symbol,
        "SHORTNAME": f"{asset} {symbol}".strip(),
        "SECNAME": f"{asset} futures {symbol}".strip(),
        "ASSETCODE": asset,
        "LASTTRADEDATE": expiry,
        "MATDATE": record.get("expiration_date") or expiry,
        "LOTSIZE": record.get("lot_size"),
        "MINSTEP": record.get("tick_size"),
        "STEPPRICE": record.get("tick_value"),
        "BUYDEPO": record.get("initial_margin"),
        "_SNAPSHOT_SOURCE": record.get("source", "moex_instrument_specs_snapshot"),
    }
