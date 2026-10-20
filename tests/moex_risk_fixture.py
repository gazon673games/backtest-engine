"""Small recorded NCC response and deterministic requests for risk-history checks."""

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.risk_history import RiskHistory

FIXTURE = Path(__file__).parent / "fixtures/moex_cny_risk_limits_20260112.json"


def payload():
    return json.loads(FIXTURE.read_text("utf-8"))


def rate_row(**overrides):
    table = payload()["limits"]
    return {
        **dict(zip(table["columns"], table["data"][0], strict=True)),
        "source_sha256": "a" * 64,
        **overrides,
    }


def history(*rows):
    return RiskHistory("CNY", tuple(rows or [rate_row()]), ())


def request(tmp_path, **overrides):
    return DatasetBuildRequest(
        **{
            "adapter_id": "moex",
            "engine": "futures",
            "market": "forts",
            "board": "RFUD",
            "security": "CR",
            "from_date": "2026-01-12",
            "till_date": "2026-01-12",
            "cache_dir": tmp_path / "cache",
            "rules_enforcement": "audit",
            **overrides,
        }
    )


def frame():
    return pl.DataFrame({"ts": [datetime(2026, 1, 12, 15)], "active_contract": ["CRH6"]})


def registry():
    return {
        "instruments": [
            {
                "symbol": "CRH6",
                "contract_multiplier": 1000,
                "lot_size": 1,
                "price_currency": "RUB",
                "pnl_currency": "RUB",
                "tick_size": 0.001,
                "tick_value": 1,
                "initial_margin": 999,
                "metadata": {"assetcode": "CNY"},
            }
        ]
    }


def audit():
    return {"status": "approved", "warnings": [], "violations": []}
