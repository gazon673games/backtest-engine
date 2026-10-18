"""Approved recorded market inputs for native robustness-protocol integration tests."""

import json

from crypto_return_fixture import SCRIPTED_ORDERS, write_return_rules
from real_crypto_fixture import TARGETS, RecordedSolTransport
from strategy_sdk.catalog_sources import snapshot_from_files

from backtest_engine.adapters.binance.adapter import BinanceAdapter
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.research.robustness.provenance import write_json
from backtest_engine.workbench.datasets import register_workbench_dataset
from backtest_engine.workbench.models import WorkbenchRunRequest


def protocol_fixture(tmp_path, product="spot"):
    build = DatasetBuildRequest(
        adapter_id="binance",
        engine=product,
        market="crypto",
        security="SOLUSDT",
        from_date="2024-04-01",
        till_date="2024-04-01",
        timeframe="60m",
        rules_path=write_return_rules(tmp_path, product),
        rules_enforcement="strict",
        history_source="rest",
        gap_policy="reject",
        output_root=tmp_path / "datasets",
    )
    dataset = BinanceAdapter(RecordedSolTransport()).build_dataset(build)
    dataset = register_workbench_dataset(tmp_path / "runs", build, dataset)
    snapshot = snapshot_from_files(
        {"scripted_crypto_return_fixture": SCRIPTED_ORDERS}, "scripted_crypto_return_fixture"
    )
    request = WorkbenchRunRequest(
        dataset_path=dataset.dataset_path,
        quality_approval_path=dataset.approval_path,
        catalog_path=dataset.catalog_path,
        dataset_id=dataset.dataset_id,
        catalog_enforcement="strict",
        instrument_registry_path=dataset.instrument_registry_path,
        calendar_path=dataset.calendar_path,
        script_source="SDK_CATALOG_SNAPSHOT = " + repr(snapshot) + "\n",
        script_language="python",
        params={"targets_json": json.dumps(TARGETS)},
        initial_capital=1000,
        base_currency="USDT",
        leverage=1 if product == "spot" else 5,
        from_date="2024-04-01",
        execution_timing="signal_close_to_next_bar_open_to_close",
        end_of_data_position_policy="reject",
    )
    write_json(tmp_path / "request.json", request.model_dump(mode="json"))
    config = {
        "name": "Recorded real SOL protocol",
        "base_request": "request.json",
        "output_root": "out",
        "split_date": "2024-04-02",
        "bootstrap_iterations": 20,
        "selection_bias_iterations": 20,
        "workers": 2,
        "cases": [
            {"name": "baseline", "group": "neighbors"},
            {"name": "inactive", "group": "neighbors", "params": {"targets_json": "{}"}},
            {"name": "high_cost", "group": "cost", "request_overrides": {"slippage_bps": 10}},
        ],
    }
    write_json(tmp_path / "protocol.json", config)
    return tmp_path / "protocol.json"
