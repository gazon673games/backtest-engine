"""Resolves current and historical MOEX futures contracts for a build."""

from __future__ import annotations

from typing import Any

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.catalog import list_moex_instrument_rows
from backtest_engine.adapters.moex.iss.contract_metadata import load_contract_metadata
from backtest_engine.adapters.moex.iss.contract_selection import expiry_date, futures_root
from backtest_engine.adapters.moex.iss.raw_snapshot_cache import snapshot_contract_rows
from backtest_engine.adapters.moex.iss.snapshot_catalog import snapshot_instrument_rows


def contracts_for_root(request: DatasetBuildRequest) -> list[dict[str, Any]]:
    rows = registry_rows(request)
    root = request.security.upper()
    registry_contracts = [row for row in rows if futures_root(str(row.get("SECID", ""))) == root]
    contracts = merge_contract_rows(
        registry_contracts,
        snapshot_contract_rows(request, root),
    )
    contracts = [
        load_contract_metadata(row, request) if expiry_date(row) is None else row
        for row in contracts
    ]
    missing = [str(row["SECID"]) for row in contracts if expiry_date(row) is None]
    if missing:
        raise ValueError(
            "MOEX contract expiry metadata is missing for "
            + ", ".join(missing)
            + "; candle download bounds cannot determine expiry"
        )
    if not contracts:
        raise ValueError(f"MOEX futures root has no contracts with expiry: {request.security}")
    ordered = sorted(contracts, key=lambda row: expiry_date(row) or "9999-12-31")
    return relevant_contracts(ordered, request)


def registry_rows(request: DatasetBuildRequest) -> list[dict[str, Any]]:
    rows = (
        snapshot_instrument_rows(
            engine=request.engine,
            market=request.market,
            board=request.board,
        )
        if request.cache_policy not in {"network", "refresh"}
        else []
    )
    if rows:
        return rows
    return list_moex_instrument_rows(
        engine=request.engine,
        market=request.market,
        board=request.board,
        cache_policy=request.cache_policy,
        cache_dir=request.cache_dir,
    )


def merge_contract_rows(
    registry_contracts: list[dict[str, Any]],
    observed_contracts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {}
    for observed in observed_contracts:
        symbol = str(observed["SECID"])
        merged[symbol] = {
            "SECID": symbol,
            "SHORTNAME": symbol,
            "SECNAME": f"Observed MOEX futures contract {symbol}",
            **observed,
        }
    for registered in registry_contracts:
        symbol = str(registered.get("SECID", ""))
        merged[symbol] = {**merged.get(symbol, {}), **registered}
    return list(merged.values())


def relevant_contracts(
    contracts: list[dict[str, Any]],
    request: DatasetBuildRequest,
) -> list[dict[str, Any]]:
    active = [row for row in contracts if (expiry_date(row) or "") >= request.from_date]
    if request.roll_rule == "volume":
        return active
    through_end = [row for row in active if (expiry_date(row) or "") <= request.till_date]
    next_contract = next(
        (row for row in active if (expiry_date(row) or "") > request.till_date),
        None,
    )
    if next_contract is not None:
        through_end.append(next_contract)
    return through_end
