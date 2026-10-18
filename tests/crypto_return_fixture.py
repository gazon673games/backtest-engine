"""Offline SOL market and scripted SDK orders for independent return oracles."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from binance_fixture import exchange_info

START = datetime(2024, 4, 1, tzinfo=UTC)
TRADE_BARS = (
    (20, 20),
    (20, 24),
    (24, 40),
    (40, 38),
    (38, 10),
    (10, 12),
    (12, 30),
    (30, 32),
)
MARK_CLOSES = (20, 25, 39, 37, 11, 13, 31, 33)
FUNDING_RATES = {0: "0.005", 1: "0.001", 3: "0.002", 5: "-0.001", 7: "0.005"}

# This test driver prescribes orders, not a reusable trading algorithm. It does
# not calculate expected PnL or access prices after the current decision.
SCRIPTED_ORDERS = """import json
from strategy_sdk import (
    DataRequirements, ParameterSpec, StrategyMetadata,
    flat_decision, hold_decision, long_decision,
)

STRATEGY_METADATA = StrategyMetadata("scripted_crypto_return_fixture", params=(
    ParameterSpec("targets_json", "str", default='{"0":"2.5","2":"3.75","4":"8.75","6":"0"}'),
    ParameterSpec("timeframe", "str", default="M60"),
    ParameterSpec("schedule_key", "str", default="hour", choices=("hour", "timestamp")),
))

def data_requirements(params):
    return DataRequirements(params["timeframe"], 1, timezone="UTC", missing_decision_policy="carry")

def initialize(params):
    return {"confirmed_fills": 0}

def generate_signal(ctx, params):
    features = {
        "confirmedFills": ctx.state["confirmed_fills"],
        "knownEquity": str(ctx.account.equity),
        "knownQuantity": str(ctx.position_quantity),
        "currency": ctx.account.currency,
        "quantityStep": str(ctx.instrument.quantity_step),
    }
    key = (
        ctx.current_ts.isoformat()
        if params["schedule_key"] == "timestamp" else str(ctx.current_ts.hour)
    )
    quantity = json.loads(params["targets_json"]).get(key)
    if quantity is None:
        return hold_decision(reason="fixture_hold", features=features)
    if quantity == "0":
        return flat_decision(reason="fixture_close", features=features)
    return long_decision(reason="fixture_buy", contracts=quantity, features=features)

def on_fill(state, fill):
    state["confirmed_fills"] += 1
"""


class SolReturnTransport:
    """Preserve Binance bar/funding boundaries while making network use impossible."""

    def get(self, product: str, path: str, params: dict[str, str | int] | None = None) -> Any:
        if path.endswith("exchangeInfo"):
            payload = exchange_info(product)
            payload["symbols"][0].update(symbol="SOLUSDT", baseAsset="SOL")
            return payload
        assert params is not None
        if path.endswith("fundingRate"):
            rows = [
                {
                    "symbol": "SOLUSDT",
                    "fundingTime": int((START + timedelta(hours=hour, minutes=30)).timestamp())
                    * 1000,
                    "fundingRate": rate,
                    "markPrice": str(MARK_CLOSES[hour]),
                }
                for hour, rate in FUNDING_RATES.items()
            ]
            return [
                row
                for row in rows
                if params["startTime"] <= row["fundingTime"] <= params["endTime"]
            ][: params["limit"]]
        assert path.endswith("klines") or path.endswith("markPriceKlines"), path
        rows = []
        for hour, (opened, closed) in enumerate(TRADE_BARS):
            stamp = int((START + timedelta(hours=hour)).timestamp()) * 1000
            if not params["startTime"] <= stamp <= params["endTime"]:
                continue
            if path.endswith("markPriceKlines"):
                opened, closed = MARK_CLOSES[max(0, hour - 1)], MARK_CLOSES[hour]
            rows.append(
                [
                    stamp,
                    str(opened),
                    str(max(opened, closed)),
                    str(min(opened, closed)),
                    str(closed),
                    "10000",
                    stamp + 3_599_999,
                    str(closed * 10000),
                    100,
                    "5000",
                    str(closed * 5000),
                    "0",
                ]
            )
        return rows[: params["limit"]]


def write_return_rules(
    root: Path, product: str, *, from_date: str = "2024-04-01", till_date: str = "2024-04-01"
) -> Path:
    snapshot = {
        "valid_from": from_date,
        "valid_to": till_date,
        "tick_size": 0.01,
        "quantity_step": 0.001,
        "minimum_quantity": 0.001,
        "minimum_notional": 5.0,
        "maker_fee_bps": 2.0,
        "taker_fee_bps": 10.0,
        "liquidation_fee_bps": 10.0,
    }
    if product == "perpetual":
        snapshot["margin_tiers"] = [
            {
                "notional_floor": 0.0,
                "notional_cap": None,
                "maintenance_margin_rate": 0.005,
                "max_leverage": 10.0,
            }
        ]
    path = root / f"sol-{product}-rules.json"
    path.write_text(
        json.dumps(
            {"venue": "binance", "product": product, "symbol": "SOLUSDT", "snapshots": [snapshot]}
        ),
        encoding="utf-8",
    )
    return path
