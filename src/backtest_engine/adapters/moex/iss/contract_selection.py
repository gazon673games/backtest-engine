"""Select contracts and audit rolls using the raw MOEX ISS response vocabulary."""

from __future__ import annotations

import re
from typing import Any


def futures_root(secid: str) -> str:
    match = re.match(r"^([A-Za-z]+)[FGHJKMNQUVXZ]\d$", secid)
    return match.group(1).upper() if match else secid.upper()


def expiry_date(row: dict[str, Any]) -> str | None:
    for key in ("LASTTRADEDATE", "lasttradedate", "MATDATE", "matdate"):
        value = row.get(key)
        if value:
            return str(value)[:10]
    return None


def nearest_rows(
    rows: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    *,
    rule: str = "expiry",
) -> list[dict[str, Any]]:
    normalized_rule = normalize_roll_rule(rule)
    expiry_by_symbol = {str(row["SECID"]): expiry_date(row) for row in contracts}
    trade_dates = sorted({row_timestamp(row)[:10] for row in rows})
    daily_leaders = expiry_leaders_by_date(trade_dates, expiry_by_symbol)
    if normalized_rule == "volume":
        # A day's total volume is only known after that trading day closes.
        observed = volume_leaders_by_date(rows, expiry_by_symbol)
        daily_leaders.update(lagged_volume_leaders(trade_dates, observed))
    return [
        row
        for row in sorted(rows, key=row_timestamp)
        if str(row.get("SECID")) == daily_leaders.get(row_timestamp(row)[:10])
    ]


def expiry_leaders_by_date(
    trade_dates: list[str],
    expiry_by_symbol: dict[str, str | None],
) -> dict[str, str]:
    contracts = sorted(
        (expiry, symbol) for symbol, expiry in expiry_by_symbol.items() if expiry is not None
    )
    if not contracts:
        return {}
    leaders = {}
    for trade_date in trade_dates:
        live = next((symbol for expiry, symbol in contracts if expiry >= trade_date), None)
        leaders[trade_date] = live or contracts[-1][1]
    return leaders


def row_timestamp(row: dict[str, Any]) -> str:
    return str(row.get("begin") or row.get("ts"))


def volume_leaders_by_date(
    rows: list[dict[str, Any]],
    expiry_by_symbol: dict[str, str | None],
) -> dict[str, str]:
    totals: dict[str, dict[str, float]] = {}
    for row in rows:
        volume = volume_value(row)
        symbol = str(row.get("SECID", ""))
        trade_date = str(row.get("begin") or row.get("ts"))[:10]
        expiry = expiry_by_symbol.get(symbol)
        if volume is None or expiry is None or expiry < trade_date:
            continue
        daily = totals.setdefault(trade_date, {})
        daily[symbol] = daily.get(symbol, 0.0) + volume

    leaders: dict[str, str] = {}
    minimum_expiry = ""
    for trade_date, daily in sorted(totals.items()):
        eligible = [
            symbol
            for symbol in daily
            if (expiry_by_symbol.get(symbol) or "") >= max(trade_date, minimum_expiry)
        ]
        if not eligible:
            continue
        leader = max(
            eligible,
            key=lambda symbol: (
                daily[symbol],
                -(int(re.sub(r"\D", "", expiry_by_symbol[symbol] or "99991231"))),
            ),
        )
        leaders[trade_date] = leader
        minimum_expiry = expiry_by_symbol.get(leader) or minimum_expiry
    return leaders


def volume_value(row: dict[str, Any]) -> float | None:
    for key in ("volume", "VOLUME", "vol"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def lagged_volume_leaders(
    trade_dates: list[str],
    observed: dict[str, str],
) -> dict[str, str]:
    """Make the completed day D volume signal effective on trading day D+1."""
    lagged: dict[str, str] = {}
    previous_leader: str | None = None
    for trade_date in trade_dates:
        if previous_leader is not None:
            lagged[trade_date] = previous_leader
        previous_leader = observed.get(trade_date, previous_leader)
    return lagged


def chain_audit(
    contracts: list[dict[str, Any]],
    *,
    roll_rule: str,
    selected_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expiries = [expiry for row in contracts if (expiry := expiry_date(row)) is not None]
    normalized_rule = normalize_roll_rule(roll_rule)
    symbols = [str(row.get("SECID", "")) for row in contracts]
    events = selected_roll_events(selected_rows or [])
    return {
        "roll_rule": normalized_rule,
        "volume_signal_lag_trading_days": 1 if normalized_rule == "volume" else 0,
        "contract_count": len(contracts),
        "first_expiry": min(expiries) if expiries else None,
        "last_expiry": max(expiries) if expiries else None,
        "symbols": symbols,
        "contract_expiries": [
            {
                "symbol": str(row["SECID"]),
                "last_trade_date": row.get("LASTTRADEDATE"),
                "expiration_date": row.get("MATDATE"),
                "selection_expiry": expiry_date(row),
                **row.get("_EXPIRY_METADATA", {}),
            }
            for row in contracts
        ],
        "roll_count": len(events),
        "roll_events": events[:100],
    }


def selected_roll_events(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    events = []
    previous = None
    for row in sorted(rows, key=row_timestamp):
        symbol = str(row.get("SECID", ""))
        if previous is not None and symbol != previous:
            events.append(
                {
                    "ts": str(row.get("begin") or row.get("ts")),
                    "from_symbol": previous,
                    "to_symbol": symbol,
                }
            )
        previous = symbol
    return events


def normalize_roll_rule(value: str) -> str:
    rule = value.strip().lower()
    if rule not in {"expiry", "volume"}:
        raise ValueError(f"Invalid nearest futures roll_rule: {value}")
    return rule
