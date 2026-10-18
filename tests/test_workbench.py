from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.adapters.models import AdapterItem
from backtest_engine.adapters.registry import get_adapter
from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.javascript.runtime import JsIntradayStrategy, js_context_payload
from backtest_engine.workbench.compare import compare_workbench_runs
from backtest_engine.workbench.history import history_path_for, load_workbench_history
from backtest_engine.workbench.live_progress import RunProgressReporter
from backtest_engine.workbench.models import WorkbenchRunRequest
from backtest_engine.workbench.progress import read_progress, write_progress
from backtest_engine.workbench.runner import run_workbench_backtest
from backtest_engine.workbench.server import get_payload
from backtest_engine.workbench.strategies import list_saved_strategies, save_strategy


def test_workbench_runs_js_source_and_writes_report_history(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for JS strategy runtime tests")
    dataset_path = write_intraday_dataset(tmp_path)
    request = WorkbenchRunRequest(
        dataset_path=dataset_path,
        quality_approval_required=False,
        allow_legacy_bar_timestamps=True,
        script_source=js_strategy_source(),
        params={"min_session_bars": "1", "max_session_bars": "10", "enabled": True},
        from_date="2024-01-02",
        end_of_data_position_policy="mark",
        output_root=tmp_path / "runs",
        run_id="fixture_run",
    )

    response = run_workbench_backtest(request)
    history = load_workbench_history(history_path_for(request.output_root))

    assert response.status == "completed"
    assert response.report["metrics"]["strategy_call_count"] == 1
    assert response.report["series"]["candles"]
    assert response.report["series"]["equity"]
    assert response.report["series"]["plots"]["session_sma"]
    assert response.report["tables"]["decision_actions"][0]["reason"] == "fixture"
    assert response.report["tables"]["decision_actions"][0]["decision_time"] == "10:10"
    assert response.report["tables"]["decision_actions"][0]["target_contracts"] == 1.0
    assert history["runs"][0]["run_id"] == "fixture_run"
    compared = compare_workbench_runs(request.output_root, ["fixture_run"])
    assert compared["runs"][0]["run_id"] == "fixture_run"


def test_workbench_run_uses_submitted_strategy_params(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for JS strategy runtime tests")
    request = WorkbenchRunRequest(
        dataset_path=write_intraday_dataset(tmp_path),
        quality_approval_required=False,
        allow_legacy_bar_timestamps=True,
        script_source=js_strategy_source(),
        params={"min_session_bars": "1", "max_session_bars": "10", "enabled": False},
        from_date="2024-01-02",
        output_root=tmp_path / "runs",
    )

    response = run_workbench_backtest(request)

    assert response.report["tables"]["decision_actions"][0]["reason"] == "disabled"


def test_js_strategy_timeout_is_reported(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for JS strategy runtime tests")
    plugin_path = tmp_path / "loop.js"
    plugin_path.write_text(
        "function onDecision(ctx, params) { while (true) {} }\n",
        encoding="utf-8",
    )
    strategy = JsIntradayStrategy(plugin_path, timeout_ms=50)
    frame = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 2, 10, 0), datetime(2024, 1, 2, 10, 5)],
            "timeframe": ["5m", "5m"],
            "close": [10.0, 11.0],
        }
    )
    ctx = IntradayStrategyContext(frame, end_index=1)

    try:
        with pytest.raises(StrategyPluginError, match="timed out"):
            strategy.execute(ctx, {})
    finally:
        strategy.close()


def test_js_context_honors_declared_history_and_columns() -> None:
    frame = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 2, 10, 0) + timedelta(minutes=5 * i) for i in range(4)],
            "timeframe": ["5m"] * 4,
            "close": [10.0, 11.0, 12.0, 13.0],
            "unused_feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    ctx = IntradayStrategyContext(frame, end_index=3)

    payload = js_context_payload(
        ctx,
        history_bars=2,
        data_columns=("ts", "close"),
    )

    assert payload["history"]["length"] == 2
    assert set(payload["history"]["columns"]) == {"ts", "close"}
    assert "unused_feature" not in payload["history"]["columns"]


def test_adapter_registry_exposes_moex_through_common_contract() -> None:
    adapter = get_adapter("moex")

    assert adapter.id == "moex"
    assert adapter.info().name == "MOEX ISS"


def test_server_routes_adapter_catalog_through_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAdapter:
        def engines(self) -> list[AdapterItem]:
            return [AdapterItem(id="demo", name="Demo engine")]

    monkeypatch.setattr(
        "backtest_engine.workbench.api.get_adapter",
        lambda adapter_id: FakeAdapter() if adapter_id == "fake" else None,
    )

    payload = get_payload(tmp_path, "/api/adapters/fake/engines", "")

    assert payload["items"] == [{"id": "demo", "name": "Demo engine", "metadata": {}}]


def test_saved_strategy_metadata_and_progress(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for JS strategy runtime tests")
    strategy = save_strategy(
        tmp_path / "strategies",
        {
            "name": "fixture",
            "source": js_strategy_source(),
            "tags": ["test"],
            "params": {"min_session_bars": "3", "max_session_bars": "9", "enabled": False},
        },
    )
    progress = write_progress(
        tmp_path / "runs",
        "r1",
        status="running",
        stage="backtest",
        message="running",
        pct=0.5,
    )

    assert strategy["metadata"]["name"] == "workbench_fixture"
    assert strategy["metadata"]["params"][0]["name"] == "min_session_bars"
    assert strategy["metadata"]["timeframe"] == "5m"
    assert strategy["metadata"]["decision_times"] == ["10:10"]
    assert strategy["metadata"]["history_bars"] == 50
    assert strategy["metadata"]["data_columns"] == [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timeframe",
        "active_contract",
    ]
    assert strategy["params"] == {
        "min_session_bars": 3,
        "max_session_bars": 9,
        "enabled": False,
    }
    assert read_progress(tmp_path / "runs", "r1")["stage"] == progress["stage"]


def test_new_strategies_with_the_same_source_get_distinct_ids(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for JS strategy runtime tests")
    root = tmp_path / "strategies"

    first = save_strategy(root, {"name": "fixture", "source": js_strategy_source()})
    second = save_strategy(root, {"name": "fixture", "source": js_strategy_source()})
    updated = save_strategy(
        root,
        {"id": first["id"], "name": "fixture updated", "source": js_strategy_source()},
    )

    assert first["id"] == "fixture"
    assert second["id"] == "fixture-2"
    assert updated["id"] == first["id"]
    assert updated["version"] == 2
    assert len(list_saved_strategies(root)) == 2


def test_run_progress_reporter_publishes_live_equity_and_decisions(tmp_path: Path) -> None:
    reporter = RunProgressReporter(tmp_path / "runs", "live_run")
    reporter(
        1,
        2,
        {
            "ts": datetime(2024, 1, 2, 10, 0),
            "fill_ts": datetime(2024, 1, 2, 10, 5),
            "equity": 1.01,
            "equity_cash": 1010.0,
            "strategy_called": True,
            "missing_decision": False,
            "target_position": 1.0,
            "target_contracts": 2.0,
            "contract_delta": 2.0,
            "filled_contract_delta": 2.0,
            "order_filled": True,
            "strategy_reason": "entry",
            "signal_reason": None,
        },
    )

    progress = read_progress(tmp_path / "runs", "live_run")
    assert progress["processed_bars"] == 1
    assert progress["total_bars"] == 2
    assert progress["live"]["equity"][0]["equity_cash"] == 1010.0
    assert progress["live"]["decisions"][0]["reason"] == "entry"
    assert progress["live"]["decisions"][0]["fill_ts"] == "2024-01-02T10:05:00"
    assert progress["live"]["decisions"][0]["filled_contract_delta"] == 2.0


def write_intraday_dataset(tmp_path: Path) -> Path:
    timestamps = [datetime(2024, 1, 2, 10, 0) + timedelta(minutes=5 * i) for i in range(5)]
    closes = [10.0, 11.0, 12.0, 11.5, 12.5]
    dataset_path = tmp_path / "intraday.parquet"
    pl.DataFrame(
        {
            "source": ["fixture"] * len(timestamps),
            "exchange": ["MOEX"] * len(timestamps),
            "engine": ["futures"] * len(timestamps),
            "market": ["forts"] * len(timestamps),
            "board": ["RFUD"] * len(timestamps),
            "symbol": ["CRH4"] * len(timestamps),
            "continuous_symbol": ["CNYRUB_NEAREST_5M"] * len(timestamps),
            "timeframe": ["5m"] * len(timestamps),
            "ts": timestamps,
            "open": closes,
            "high": [value + 0.1 for value in closes],
            "low": [value - 0.1 for value in closes],
            "close": closes,
            "volume": [1000.0] * len(timestamps),
            "value": [100000.0] * len(timestamps),
            "currency": ["RUB"] * len(timestamps),
            "active_contract": ["CRH4"] * len(timestamps),
            "source_contract_symbol": ["CRH4"] * len(timestamps),
            "dataset_type": ["nearest_contract"] * len(timestamps),
            "adjustment_mode": ["unadjusted"] * len(timestamps),
        }
    ).write_parquet(dataset_path)
    return dataset_path


def js_strategy_source() -> str:
    return """
globalThis.STRATEGY_METADATA = {
  name: "workbench_fixture",
  timeframe: "5m",
  schedule: "times",
  decision_times: ["10:10"],
  missing_decision_policy: "close",
  history_bars: 50,
  data_columns: ["ts", "open", "high", "low", "close", "volume", "timeframe", "active_contract"],
  params: [
    { name: "min_session_bars", param_type: "int", default: 1, min_value: 1 },
    { name: "max_session_bars", param_type: "int", default: 10, min_value: 1 },
    { name: "enabled", param_type: "bool", default: true },
  ],
};

function onDecision(ctx, params) {
  console.log("decision", ctx.time);
  if (!params.enabled) return { target: 0, contracts: 0, reason: "disabled" };
  const closes = ctx.session.values("close", { from: "10:00" });
  const sma = math.avg(closes);
  plot.line("session_sma", sma);
  let target = 0;
  if (ctx.bar.close > sma) target = 1;
  if (ctx.bar.close < sma) target = -1;
  return { target, contracts: 1, reason: "fixture", debug: { sma, count: closes.length } };
}
""".lstrip()
