"""Downloads nearest-futures source contracts in publishable chronological slices."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

from backtest_engine.adapters.build_events import DatasetBuildObserver
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.candle_transport import RawRowsCallback
from backtest_engine.adapters.moex.iss.contract_catalog import contracts_for_root
from backtest_engine.adapters.moex.iss.contract_selection import expiry_date, nearest_rows
from backtest_engine.adapters.moex.sessions import MOEX_TIMEZONE

SecurityLoader = Callable[
    [DatasetBuildRequest, RawRowsCallback | None],
    list[dict[str, Any]],
]


def download_nearest_futures_candles(
    request: DatasetBuildRequest,
    *,
    load_security: SecurityLoader,
    on_rows: RawRowsCallback | None,
    observer: DatasetBuildObserver | None,
) -> list[dict[str, Any]]:
    contracts = contracts_for_root(request)
    live_symbol = live_contract_symbol(contracts)
    all_rows: list[dict[str, Any]] = []
    skipped = []
    previous_expiry: str | None = None
    for index, contract in enumerate(contracts):
        child = child_request(
            request,
            contract,
            previous_expiry=previous_expiry,
            live_symbol=live_symbol,
        )
        previous_expiry = expiry_date(contract)
        if child is None:
            continue
        try:
            callback = on_rows if request.roll_rule == "expiry" else None
            all_rows.extend(load_security(child, callback))
        except ValueError as exc:
            if "no candles" not in str(exc).lower():
                raise
            skipped.append(str(contract["SECID"]))
        publish_contract_progress(observer, request, contract, index, len(contracts))
    if not all_rows:
        suffix = f"; checked contracts: {', '.join(skipped)}" if skipped else ""
        raise ValueError(f"MOEX returned no candles for {request.security}{suffix}")
    selected = nearest_rows(all_rows, contracts, rule=request.roll_rule)
    if request.roll_rule == "volume" and on_rows is not None:
        on_rows(selected)
    return selected


def child_request(
    request: DatasetBuildRequest,
    contract: dict[str, Any],
    *,
    previous_expiry: str | None,
    live_symbol: str | None,
) -> DatasetBuildRequest | None:
    update: dict[str, Any] = {
        "security": str(contract["SECID"]),
        "instrument_mode": "security",
        "refresh_live_tail": (request.refresh_live_tail and str(contract["SECID"]) == live_symbol),
    }
    if request.roll_rule == "expiry":
        active_from = request.from_date
        if previous_expiry is not None:
            active_from = max(request.from_date, next_date(previous_expiry))
        active_till = min(request.till_date, expiry_date(contract) or request.till_date)
        if active_from > active_till:
            return None
        update.update({"from_date": active_from, "till_date": active_till})
    return request.model_copy(update=update)


def live_contract_symbol(contracts: list[dict[str, Any]]) -> str | None:
    today = datetime.now(MOEX_TIMEZONE).date().isoformat()
    live = [row for row in contracts if (expiry_date(row) or "") >= today]
    if not live:
        return None
    return str(min(live, key=lambda row: expiry_date(row) or "9999-12-31")["SECID"])


def next_date(value: str) -> str:
    return (date.fromisoformat(value) + timedelta(days=1)).isoformat()


def publish_contract_progress(
    observer: DatasetBuildObserver | None,
    request: DatasetBuildRequest,
    contract: dict[str, Any],
    index: int,
    total: int,
) -> None:
    if observer is None:
        return
    completed = index + 1
    observer.progress(
        stage="dataset_download",
        message=f"loaded {request.security} contract {completed}/{total}: {contract['SECID']}",
        pct=0.15 + 0.5 * completed / max(1, total),
    )
