from __future__ import annotations

from pathlib import Path
from typing import Any

from backtest_engine.adapters.models import AdapterItem
from backtest_engine.adapters.moex.iss.client import ISS_BASE_URL, fetch_iss_json, iss_table_rows
from backtest_engine.adapters.moex.iss.contract_selection import expiry_date, futures_root
from backtest_engine.adapters.moex.iss.snapshot_catalog import snapshot_instrument_rows
from backtest_engine.adapters.moex.pipeline.paths import optional_string, quote

FUTURES_ROOT_NAMES = {
    "CR": "CNY/RUB - Chinese yuan / Russian ruble",
    "SI": "USD/RUB - US dollar / Russian ruble",
}
INSTRUMENT_COLUMNS = (
    "SECID,SHORTNAME,SECNAME,ASSETCODE,LASTTRADEDATE,MATDATE,"
    "LOTVOLUME,LOTSIZE,MINSTEP,STEPPRICE,INITIALMARGIN,BUYDEPO,BUYSELLFEE,SCALPERFEE"
)


def list_moex_engines() -> list[AdapterItem]:
    rows = iss_table_rows(fetch_iss_json(f"{ISS_BASE_URL}/engines.json"), "engines")
    return [adapter_item(row, "name", "title") for row in rows]


def list_moex_markets(engine: str = "futures") -> list[AdapterItem]:
    url = f"{ISS_BASE_URL}/engines/{quote(engine)}/markets.json"
    rows = iss_table_rows(fetch_iss_json(url), "markets")
    return [adapter_item(row, "market_name", "market_title") for row in rows]


def list_moex_boards(engine: str = "futures", market: str = "forts") -> list[AdapterItem]:
    url = f"{ISS_BASE_URL}/engines/{quote(engine)}/markets/{quote(market)}/boards.json"
    rows = iss_table_rows(fetch_iss_json(url), "boards")
    return [adapter_item(row, "boardid", "title") for row in rows]


def list_moex_instruments(
    *,
    engine: str = "futures",
    market: str = "forts",
    board: str = "RFUD",
    query: str = "",
    limit: int = 200,
) -> list[dict[str, Any]]:
    rows = list_moex_instrument_rows(engine=engine, market=market, board=board)
    query_lower = query.strip().lower()
    filtered = [row for row in rows if instrument_matches(row, query_lower)]
    return [instrument_payload(row, engine, market, board) for row in filtered[:limit]]


def list_moex_futures_roots(
    *,
    engine: str = "futures",
    market: str = "forts",
    board: str = "RFUD",
    query: str = "",
    limit: int = 200,
) -> list[dict[str, Any]]:
    rows = list_moex_instrument_rows(engine=engine, market=market, board=board)
    chains: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        root = futures_root(str(row.get("SECID", "")))
        if root:
            chains.setdefault(root, []).append(row)
    items = [
        chain_payload(root, contracts, engine, market, board) for root, contracts in chains.items()
    ]
    query_lower = query.strip().lower()
    filtered = [item for item in items if chain_matches(item, query_lower)]
    return sorted(filtered, key=lambda item: item["symbol"])[:limit]


def list_moex_snapshot_futures_roots(
    *,
    engine: str = "futures",
    market: str = "forts",
    board: str = "RFUD",
    query: str = "",
    limit: int = 200,
) -> list[dict[str, Any]]:
    rows = snapshot_instrument_rows(engine=engine, market=market, board=board)
    chains: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        root = futures_root(str(row.get("SECID", "")))
        if root:
            chains.setdefault(root, []).append(row)
    items = [
        chain_payload(root, contracts, engine, market, board) for root, contracts in chains.items()
    ]
    query_lower = query.strip().lower()
    filtered = [item for item in items if chain_matches(item, query_lower)]
    return sorted(filtered, key=lambda item: item["symbol"])[:limit]


def list_moex_instrument_rows(
    *,
    engine: str = "futures",
    market: str = "forts",
    board: str = "RFUD",
    cache_policy: str | None = None,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    url = (
        f"{ISS_BASE_URL}/engines/{quote(engine)}/markets/{quote(market)}"
        f"/boards/{quote(board)}/securities.json"
    )
    payload = fetch_iss_json(
        url,
        {
            "iss.meta": "off",
            "iss.only": "securities",
            "securities.columns": INSTRUMENT_COLUMNS,
        },
        cache_policy=cache_policy,
        cache_dir=cache_dir,
    )
    return iss_table_rows(payload, "securities")


def adapter_item(row: dict[str, Any], id_key: str, name_key: str) -> AdapterItem:
    item_id = str(row.get(id_key) or row.get("id") or "")
    name = str(row.get(name_key) or row.get("title") or item_id)
    return AdapterItem(id=item_id, name=name, metadata=row)


def instrument_matches(row: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    haystack = " ".join(str(row.get(key, "")) for key in ("SECID", "SHORTNAME", "SECNAME"))
    return query in haystack.lower()


def instrument_payload(row: dict[str, Any], engine: str, market: str, board: str) -> dict[str, Any]:
    return {
        "symbol": str(row.get("SECID", "")),
        "root_symbol": futures_root(str(row.get("SECID", ""))),
        "short_name": optional_string(row.get("SHORTNAME")),
        "name": optional_string(row.get("SECNAME")),
        "board": board,
        "market": market,
        "engine": engine,
        "metadata": row,
    }


def chain_payload(
    root: str,
    contracts: list[dict[str, Any]],
    engine: str,
    market: str,
    board: str,
) -> dict[str, Any]:
    sorted_contracts = sorted(contracts, key=lambda row: expiry_date(row) or "9999-12-31")
    first = sorted_contracts[0]
    return {
        "symbol": root,
        "display_name": FUTURES_ROOT_NAMES.get(
            root,
            optional_string(first.get("SHORTNAME")) or f"{root} futures",
        ),
        "short_name": optional_string(first.get("SHORTNAME")),
        "name": f"{root} nearest futures",
        "board": board,
        "market": market,
        "engine": engine,
        "contract_count": len(sorted_contracts),
        "nearest_symbol": str(first.get("SECID", "")),
        "contracts": [instrument_payload(row, engine, market, board) for row in sorted_contracts],
    }


def chain_matches(item: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    haystack = " ".join(str(item.get(key, "")) for key in ("symbol", "short_name", "name"))
    return query in haystack.lower()
