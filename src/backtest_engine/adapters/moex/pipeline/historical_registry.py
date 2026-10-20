"""Attach pinned risk history or exact dated rules to each dataset's instrument registry."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.risk_history import download_risk_history
from backtest_engine.adapters.moex.pipeline.risk_estimates import margin_schedule
from backtest_engine.data.instrument_registry import InstrumentSpec


def enrich_historical_registry(
    request: DatasetBuildRequest,
    reports: Path,
    frame: pl.DataFrame,
    registry: dict[str, Any],
    audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    exact, rules_source = load_exact_rules(request.rules_path)
    dates = contract_dates(frame)
    row_assets = {
        str(row["symbol"]): str(row.get("metadata", {}).get("assetcode") or "")
        for row in registry["instruments"]
    }
    histories, failures = {}, []
    needed_assets = {
        row_assets[item["symbol"]]
        for item in registry["instruments"]
        if any(
            not any(
                rule["symbol"] == item["symbol"]
                and date.fromisoformat(rule["valid_from"])
                <= day
                <= date.fromisoformat(rule["valid_to"])
                for rule in exact
            )
            for day in dates.get(item["symbol"], [])
        )
    }
    for asset in sorted(needed_assets - {""}):
        try:
            histories[asset] = download_risk_history(asset, request, reports)
        except (OSError, ValueError) as error:
            failures.append(f"{asset}: {error}")
    instruments, missing, estimated = [], [], []
    warnings = []
    for original in registry["instruments"]:
        spec = {**original, "initial_margin": None}
        symbol = spec["symbol"]
        needed = dates.get(symbol, [])
        exact_rows = [{**spec, **item} for item in exact if item["symbol"] == symbol]
        instruments.extend(exact_rows)
        covered = {
            day
            for day in needed
            if any(
                date.fromisoformat(item["valid_from"])
                <= day
                <= date.fromisoformat(item["valid_to"])
                for item in exact_rows
            )
        }
        uncovered = [day for day in needed if day not in covered]
        if not uncovered:
            continue
        history = histories.get(row_assets[symbol])
        schedule = []
        if history is not None and request.rules_enforcement == "audit":
            schedule = margin_schedule(
                history, uncovered, spec["contract_multiplier"], carry_weekend_rates=True
            )
        spec.update(
            margin_estimate_schedule=schedule,
            rule_source="ncc_mr1_notional_estimate_v1" if schedule else "missing_historical_margin",
        )
        instruments.append(spec)
        if schedule:
            estimated.append(symbol)
        for day in uncovered:
            intervals = [
                interval for interval in schedule if interval["valid_from"][:10] == day.isoformat()
            ]
            timestamps = frame.filter(
                (pl.col("active_contract") == symbol) & (pl.col("ts").dt.date() == day)
            )["ts"]
            if not intervals or any(
                not any(
                    datetime.fromisoformat(item["valid_from"])
                    <= stamp
                    < datetime.fromisoformat(item["valid_to"])
                    for item in intervals
                )
                for stamp in timestamps
            ):
                missing.append(f"{symbol}: historical margin unavailable on {day}")
    for item in instruments:
        for field in ("tick_size", "tick_value"):
            if item.get(field) is None:
                warnings.append(f"{item['symbol']}: {field} is missing in historical metadata")
    if estimated:
        warnings.append(
            "Research margin estimate: contract close * point value * historical NCC MR1, "
            "rounded up to RUB 0.01; excludes interest/clearing scenarios and broker add-ons. "
            "Limited to the first concentration tier. "
            "Weekends use the last known trading-day rate. "
            "Exact exchange initial margin is unknown."
        )
    warnings.extend(failures)
    warnings.extend(missing)
    source = {
        "model": "ncc_mr1_notional_estimate_v1",
        "estimated_contracts": estimated,
        "missing_coverage": missing,
        "weekend_rate_policy": "last_known_trading_day_rate",
        "carried_intervals": [
            {
                "symbol": item["symbol"],
                "valid_from": interval["valid_from"],
                "source_trade_date": interval["source_trade_date"],
            }
            for item in instruments
            for interval in item.get("margin_estimate_schedule", [])
            if interval["carried_forward"]
        ],
        "sources": [source for history in histories.values() for source in history.sources],
        "exact_rules": rules_source,
        "download_failures": failures,
    }
    (reports / "risk_history").mkdir(parents=True, exist_ok=True)
    (reports / "risk_history" / "audit.json").write_text(
        json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    registry.update(instruments=instruments, historical_margin=source)
    audit.update(
        status="rejected" if audit["violations"] else "warning" if warnings else "approved",
        warnings=warnings,
        historical_margin=source,
        instrument_count=len(instruments),
    )
    return registry, audit


def load_exact_rules(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return [], None
    raw = path.read_bytes()
    payload = json.loads(raw)
    records = payload.get("instruments") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("MOEX rules_path must contain a nonempty instruments list")
    result = []
    for record in records:
        spec = InstrumentSpec.model_validate(record)
        if (
            spec.initial_margin is None
            or spec.initial_margin <= 0
            or spec.valid_from is None
            or spec.valid_to is None
            or spec.valid_to < spec.valid_from
            or spec.margin_estimate_schedule
            or any(char in spec.symbol for char in "*?[")
        ):
            raise ValueError(
                f"Exact MOEX rules for {spec.symbol} require positive initial_margin, "
                "exact symbol, valid_from and valid_to; estimates are not exact rules"
            )
        if any(
            item["symbol"] == spec.symbol
            and date.fromisoformat(item["valid_from"]) <= spec.valid_to
            and date.fromisoformat(item["valid_to"]) >= spec.valid_from
            for item in result
        ):
            raise ValueError(f"Overlapping exact MOEX rules for {spec.symbol}")
        result.append(
            {
                **record,
                "rule_source": spec.rule_source or "explicit_historical_rules",
                "valid_from": spec.valid_from.isoformat(),
                "valid_to": spec.valid_to.isoformat(),
            }
        )
    return result, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def contract_dates(frame: pl.DataFrame) -> dict[str, list[date]]:
    pairs = frame.select("active_contract", pl.col("ts").dt.date().alias("day")).unique()
    result: dict[str, list[date]] = {}
    for symbol, day in pairs.iter_rows():
        result.setdefault(symbol, []).append(day)
    return {symbol: sorted(days) for symbol, days in result.items()}
