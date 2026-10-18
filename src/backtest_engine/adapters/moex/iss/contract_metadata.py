"""Resolve missing expiry dates from cached, explicit MOEX security descriptions."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.client import ISS_BASE_URL, fetch_iss_json, iss_table_rows
from backtest_engine.adapters.moex.pipeline.paths import quote

DESCRIPTION_PARAMS = {
    "iss.meta": "off",
    "iss.only": "description",
    "description.columns": "name,value",
}


def load_contract_metadata(
    row: dict[str, Any],
    request: DatasetBuildRequest,
) -> dict[str, Any]:
    symbol = str(row["SECID"])
    url = f"{ISS_BASE_URL}/securities/{quote(symbol)}.json"
    try:
        payload = fetch_iss_json(
            url,
            DESCRIPTION_PARAMS,
            cache_policy=request.cache_policy,
            cache_dir=request.cache_dir,
        )
    except (OSError, ValueError) as exc:
        hint = (
            "; load once with cache_policy=prefer or provide a trusted instrument snapshot"
            if request.cache_policy == "offline"
            else ""
        )
        raise ValueError(
            f"MOEX contract expiry metadata is missing for {symbol}: "
            f"could not load {url}: {exc}{hint}"
        ) from exc
    description = {
        str(item.get("name", "")).upper(): item.get("value")
        for item in iss_table_rows(payload, "description")
    }
    actual_symbol = str(description.get("SECID") or "")
    if actual_symbol.upper() != symbol.upper():
        raise ValueError(
            f"MOEX contract metadata mismatch for {symbol}: "
            f"{url} returned SECID={actual_symbol or '<missing>'}"
        )
    # Listing/candle coverage describes available data, not the contractual expiry.
    last_trade = description_date(description, ("LSTTRADE", "LASTTRADEDATE"), symbol)
    expiration = description_date(description, ("LSTDELDATE", "MATDATE"), symbol)
    metadata = {
        "SHORTNAME": description.get("SHORTNAME"),
        "SECNAME": description.get("NAME") or description.get("CONTRACTNAME"),
        "ASSETCODE": description.get("ASSETCODE"),
        "LOTSIZE": description.get("LOTSIZE"),
        "LASTTRADEDATE": last_trade,
        "MATDATE": expiration,
    }
    result = {key: value for key, value in metadata.items() if value not in (None, "")}
    result.update({key: value for key, value in row.items() if value not in (None, "")})
    result["LASTTRADEDATE"] = last_trade
    result["MATDATE"] = expiration
    result["_EXPIRY_METADATA"] = {
        "source_url": url,
        "source_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "last_trade_date": last_trade,
        "expiration_date": expiration,
        "source_payload": payload,
    }
    return result


def description_date(
    description: dict[str, Any],
    keys: tuple[str, ...],
    symbol: str,
) -> str | None:
    for key in keys:
        value = description.get(key)
        if value not in (None, ""):
            try:
                return date.fromisoformat(str(value)[:10]).isoformat()
            except ValueError as exc:
                raise ValueError(f"MOEX {symbol}: invalid {key} expiry date {value!r}") from exc
    return None
