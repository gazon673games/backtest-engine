from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime
from backtest_engine.adapters.moex.pipeline.models import MoexDataPipelineConfig
from backtest_engine.adapters.moex.pipeline.runner import run_moex_data_pipeline
from backtest_engine.portfolio.multi.config import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentExecutionConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
)
from backtest_engine.portfolio.multi.execution_v3 import market_fill_result
from backtest_engine.research.benchmark import DatasetBenchmarkConfig, benchmark_dataset
from backtest_engine.research.external_validation import (
    ExternalValidationConfig,
    validate_against_statement,
)
from backtest_engine.strategies.plugins.loader import (
    generate_intraday_event_plugin_decision_from_callable,
    load_intraday_strategy_plugin,
    strategy_file_sha256,
)


def test_moex_data_pipeline_builds_nearest_continuous_and_approval(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps({"rows": raw_rows()}), encoding="utf-8")

    result = run_moex_data_pipeline(
        MoexDataPipelineConfig(
            raw_snapshot_path=raw_path,
            output_root=tmp_path / "out",
            continuous_symbol="CR_NEAREST",
            aggregate_timeframes=["5m"],
            gap_policy="audit",
        )
    )
    nearest = pl.read_parquet(result.nearest_path)
    approval = json.loads(result.approval_path.read_text(encoding="utf-8"))

    assert result.approval_status == "approved"
    assert nearest.get_column("continuous_symbol").to_list() == ["CR_NEAREST"]
    assert approval["gap_audit"]["gap_count"] == 0


def test_moex_data_pipeline_repairs_gaps_with_zero_volume_bars(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_gap.json"
    rows = raw_rows()
    rows[1]["begin"] = "2024-01-02T10:03:00"
    raw_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    result = run_moex_data_pipeline(
        MoexDataPipelineConfig(
            raw_snapshot_path=raw_path,
            output_root=tmp_path / "out_gap",
            aggregate_timeframes=["1m"],
            gap_repair_policy="zero_volume_forward_fill",
        )
    )
    normalized = pl.read_parquet(result.normalized_path)
    repair = json.loads(result.gap_repair_path.read_text(encoding="utf-8"))

    assert result.approval_status == "approved"
    assert repair["synthetic_bar_count"] == 2
    assert normalized.filter(pl.col("is_synthetic")).height == 2
    assert normalized.filter(pl.col("is_synthetic")).get_column("volume").to_list() == [0.0, 0.0]


def test_moex_quality_gate_rejects_excessive_unrepaired_gap_ratio(tmp_path: Path) -> None:
    rows = raw_rows()
    rows[1]["begin"] = "2024-01-02T10:10:00"
    raw_path = tmp_path / "raw_bad_gap.json"
    raw_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    result = run_moex_data_pipeline(
        MoexDataPipelineConfig(
            raw_snapshot_path=raw_path,
            output_root=tmp_path / "out_bad_gap",
            aggregate_timeframes=["1m"],
            gap_policy="audit",
            max_gap_ratio=0.01,
        )
    )
    approval = json.loads(result.approval_path.read_text(encoding="utf-8"))

    assert result.approval_status == "rejected"
    assert approval["gap_audit"]["missing_bar_count"] == 9
    assert approval["violations"]


def test_moex_quality_gate_rejects_excessive_roll_gap(tmp_path: Path) -> None:
    rows = raw_rows()
    rows[1]["SECID"] = "CRZ4"
    rows[1]["CLOSE"] = 130.0
    rows[1]["HIGH"] = 131.0
    raw_path = tmp_path / "raw_bad_roll.json"
    raw_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    result = run_moex_data_pipeline(
        MoexDataPipelineConfig(
            raw_snapshot_path=raw_path,
            output_root=tmp_path / "out_bad_roll",
            aggregate_timeframes=["1m"],
            max_roll_gap_pct=0.1,
        )
    )
    roll = json.loads(result.roll_audit_path.read_text(encoding="utf-8"))

    assert result.approval_status == "rejected"
    assert roll["status"] == "warning"
    assert roll["max_observed_gap_pct"] > 0.1


def test_ledger_runtime_is_equity_source_with_fx() -> None:
    runtime = BacktestLedgerRuntime(
        initial_capital=1000.0,
        base_currency="RUB",
        fx_rates={"USD/RUB": 90.0},
        start_ts=datetime(2024, 1, 2, 10, 0),
    )

    equity = runtime.apply_event_cash_flow(
        ts=datetime(2024, 1, 2, 10, 5),
        symbol="USDRUB",
        currency="USD",
        gross_pnl=1.0,
        cost_cash=0.1,
    )

    assert equity == pytest.approx(1081.0)


def test_multi_instrument_v3_partial_fill_result(tmp_path: Path) -> None:
    config = MultiInstrumentBacktestConfig.model_construct(
        legs=[],
        strategy=MultiInstrumentStrategyConfig(strategy_file=tmp_path / "s.py"),
        portfolio=MultiInstrumentPortfolioConfig(),
        execution=MultiInstrumentExecutionConfig(volume_participation_limit=0.5),
        output=MultiInstrumentOutputConfig(run_id="x"),
    )
    result = market_fill_result(
        rows=[
            {
                "ts": datetime(2024, 1, 2, 10, 0),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 2.0,
            }
        ],
        index=0,
        base_fill_price=100.0,
        previous_contracts=0.0,
        requested_contracts=3.0,
        config=config,
    )

    assert result.target_contracts == pytest.approx(1.0)
    assert result.fill_fraction == pytest.approx(1.0 / 3.0)


def test_strict_strategy_runtime_requires_metadata(tmp_path: Path) -> None:
    plugin_path = tmp_path / "plain.py"
    plugin_path.write_text("def generate_signal(ctx):\n    return 1.0\n", encoding="utf-8")

    with pytest.raises(Exception, match="STRATEGY_METADATA"):
        load_intraday_strategy_plugin(plugin_path, require_metadata=True)


def test_js_intraday_strategy_plugin_uses_low_level_context(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for JS strategy runtime tests")
    plugin_path = tmp_path / "fixture.js"
    plugin_path.write_text(
        """
globalThis.STRATEGY_METADATA = {
  name: "js_fixture",
  version: "1",
  warmup_bars: 2,
  params: [{ name: "window", param_type: "int", default: 2, min_value: 1 }],
};

function onDecision(ctx, params) {
  const closes = ctx.session.values("close", { from: "10:00" });
  const sma = math.sma(closes, params.window);
  plot.line("sma", sma);

  let target = 0;
  if (ctx.bar.close > sma) {
    target = 1;
  }
  if (ctx.bar.close < sma) {
    target = -1;
  }
  return { target, reason: "raw_if", debug: { sma, count: closes.length } };
}
""".lstrip(),
        encoding="utf-8",
    )
    generate_signal = load_intraday_strategy_plugin(plugin_path, require_metadata=True)
    from backtest_engine.strategies.intraday_api import IntradayStrategyContext

    frame = pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 2, 10, 0),
                datetime(2024, 1, 2, 10, 5),
                datetime(2024, 1, 2, 10, 10),
            ],
            "timeframe": ["5m", "5m", "5m"],
            "close": [10.0, 11.0, 12.0],
        }
    )
    ctx = IntradayStrategyContext(frame, end_index=2)
    decision = generate_intraday_event_plugin_decision_from_callable(
        generate_signal,
        ctx,
        {},
    )

    assert decision.target == pytest.approx(1.0)
    assert decision.reason == "raw_if"
    assert decision.features["sma"] == pytest.approx(11.5)
    assert decision.features["count"] == 3
    assert decision.features["plots"][0]["name"] == "sma"
    assert len(strategy_file_sha256(plugin_path)) == 64


def test_external_validation_and_benchmark(tmp_path: Path) -> None:
    events_path = tmp_path / "events.parquet"
    statement_path = tmp_path / "statement.csv"
    frame = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 2, 10, 0), datetime(2024, 1, 2, 10, 5)],
            "net_pnl": [1.0, -0.5],
            "cost_cash": [0.1, 0.2],
            "equity_cash": [1001.0, 1000.5],
            "close": [10.0, 11.0],
        }
    )
    frame.write_parquet(events_path)
    frame.write_csv(statement_path)

    validation = validate_against_statement(
        ExternalValidationConfig(events_path, statement_path, tmp_path / "validation.json")
    )
    benchmark = benchmark_dataset(DatasetBenchmarkConfig(events_path, tmp_path / "bench.json"))

    assert validation.status == "approved"
    assert benchmark.row_count == 2


def raw_rows() -> list[dict[str, object]]:
    return [
        {
            "begin": "2024-01-02T10:00:00",
            "SECID": "CRU4",
            "OPEN": 100.0,
            "HIGH": 101.0,
            "LOW": 99.0,
            "CLOSE": 100.5,
            "VOLUME": 10.0,
            "VALUE": 1000.0,
        },
        {
            "begin": "2024-01-02T10:01:00",
            "SECID": "CRU4",
            "OPEN": 100.5,
            "HIGH": 102.0,
            "LOW": 100.0,
            "CLOSE": 101.0,
            "VOLUME": 11.0,
            "VALUE": 1111.0,
        },
    ]
