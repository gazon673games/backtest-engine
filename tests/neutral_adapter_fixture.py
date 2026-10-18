"""A third, offline venue with non-crypto USD instruments and unusual sessions."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import polars as pl
from strategy_sdk.catalog_sources import snapshot_from_files

from backtest_engine.adapters.models import (
    AdapterCapabilities,
    AdapterInfo,
    AdapterItem,
    DatasetBuildRequest,
    DatasetBuildResult,
)
from backtest_engine.adapters.request_defaults import resolve_build_request
from backtest_engine.data.bar_availability import ensure_bar_availability
from backtest_engine.data.store import file_sha256
from backtest_engine.workbench.models import WorkbenchRunRequest

SCRIPT = """from strategy_sdk import (DataRequirements, StrategyMetadata,
    long_decision, flat_decision, hold_decision)
STRATEGY_METADATA = StrategyMetadata("neutral_venue_order_fixture")
def data_requirements(params):
    return DataRequirements("M5", 1, timezone="UTC")
def initialize(params):
    return {"calls": 0, "fills": 0}
def on_fill(state, fill):
    state["fills"] += 1
def generate_signal(ctx, params):
    ctx.state["calls"] += 1
    features = {"currency": ctx.account.currency, "fills": ctx.state["fills"]}
    if ctx.state["calls"] == 1:
        return long_decision(reason="fixture_buy", contracts="2.5", features=features)
    if ctx.state["calls"] == 3:
        return flat_decision(reason="fixture_sell", features=features)
    return hold_decision(reason="fixture_hold", features=features)
"""


class NeutralAdapter:
    id = "neutral-test-venue"
    products = {"cash": "spot", "dated": "future", "swap": "perpetual"}

    def info(self):
        return AdapterInfo(
            id=self.id,
            name="Neutral venue",
            description="Recorded offline fixture",
            capabilities=AdapterCapabilities(
                asset_classes=["commodity"],
                default_engine="cash",
                timeframes=["5m"],
                default_delay_minutes=2,
                supports_offline=True,
            ),
        )

    def engines(self):
        return [AdapterItem(id=key, name=key) for key in self.products]

    def markets(self, engine):
        return [AdapterItem(id="test-market", name="Test market")]

    def boards(self, engine, market):
        return [AdapterItem(id="test-board", name="Test board")]

    def instruments(self, **kwargs):
        return [{"symbol": "GOLDUSD"}]

    def build_dataset(self, request: DatasetBuildRequest, observer=None) -> DatasetBuildResult:
        request = resolve_build_request(request, self)
        kind = self.products[request.engine]
        root = request.output_root / uuid4().hex
        root.mkdir(parents=True)
        prices = [100.0, 100.0, 110.0, 120.0, 120.0, 120.0]
        start = datetime(2024, 1, 2, 1, 0)
        frame = pl.DataFrame(
            {
                "ts": [start + timedelta(minutes=5 * i) for i in range(len(prices))],
                **{
                    column: prices
                    for column in (
                        "open",
                        "high",
                        "low",
                        "close",
                        "mark_price",
                        "mark_low",
                        "mark_high",
                    )
                },
                "volume": [1000.0] * len(prices),
                "value": [100000.0] * len(prices),
                **{
                    column: [value] * len(prices)
                    for column, value in {
                        "source": "offline_fixture",
                        "exchange": self.id,
                        "engine": request.engine,
                        "market": request.market,
                        "board": request.board,
                        "symbol": "GOLDUSD",
                        "timeframe": "5m",
                        "currency": "USD",
                        "base_currency": "GOLD",
                        "quote_currency": "USD",
                        "settlement_currency": "USD",
                        "asset_class": "commodity",
                        "instrument_type": kind,
                        "dataset_type": "nearest_contract" if kind == "future" else kind,
                        "continuous_symbol": "GOLDUSD",
                        "active_contract": "GOLDUSD",
                        "source_contract_symbol": "GOLDUSD",
                        "adjustment_mode": "unadjusted",
                    }.items()
                },
            }
        )
        frame = ensure_bar_availability(frame, "5m")
        dataset = root / "prices.parquet"
        frame.write_parquet(dataset)
        spec = {
            "symbol": "GOLDUSD",
            "asset_class": "commodity",
            "instrument_type": kind,
            "contract_type": "spot" if kind == "spot" else "linear",
            "base_currency": "GOLD",
            "quote_currency": "USD",
            "settlement_currency": "USD",
            "price_currency": "USD",
            "pnl_currency": "USD",
            "quantity_step": 0.5,
            "initial_margin": 10.0,
            "exchange_fee_per_contract": 0.1 if kind == "future" else 0.0,
            "taker_fee_bps": 0.0 if kind == "future" else 10.0,
            "margin_tiers": [
                {"notional_floor": 0.0, "maintenance_margin_rate": 0.005, "max_leverage": 5.0}
            ],
        }
        files = {
            "rules.json": {"instruments": [spec]},
            "calendar.json": {
                "timezone": "UTC",
                "trading_days_per_year": 260,
                "main_session_start": "00:30",
                "main_session_end": "02:00",
                "evening_session_start": None,
                "evening_session_end": None,
            },
            "approval.json": {
                "approval_status": "approved",
                "dataset_sha256": file_sha256(dataset),
                "usable_decision_day_pct": 1.0,
                "timeframe": "5m",
                "continuous_symbol": "GOLDUSD",
            },
            "raw.json": {"prices": prices},
        }
        for name, payload in files.items():
            (root / name).write_text(json.dumps(payload), encoding="utf-8")
        manifest = {
            "files": {
                key: {"path": str(path), "sha256": file_sha256(path)}
                for key, path in {"dataset": dataset, "raw": root / "raw.json"}.items()
            }
        }
        (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return DatasetBuildResult(
            adapter_id=self.id,
            dataset_id=f"neutral-{kind}",
            dataset_path=dataset,
            output_root=root,
            approval_status="approved",
            approval_path=root / "approval.json",
            manifest_path=root / "dataset_manifest.json",
            raw_snapshot_path=root / "raw.json",
            timeframe="5m",
            row_count=len(prices),
            instrument_registry_path=root / "rules.json",
            calendar_path=root / "calendar.json",
        )


def run_request(root: Path, dataset: DatasetBuildResult, *, run_id="neutral"):
    snapshot = snapshot_from_files({"neutral_order_fixture": SCRIPT}, "neutral_order_fixture")
    return WorkbenchRunRequest(
        dataset_path=dataset.dataset_path,
        dataset_id=dataset.dataset_id,
        catalog_path=dataset.catalog_path,
        catalog_enforcement="strict",
        script_language="python",
        script_source="SDK_CATALOG_SNAPSHOT = " + repr(snapshot) + "\n",
        initial_capital=1000,
        sizing_mode="strategy_contracts",
        from_date="2024-01-02",
        execution_timing="signal_close_to_next_bar_open_to_close",
        session_filter="main",
        output_root=root / "runs",
        run_id=run_id,
    )
