from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from backtest_engine.accounting.funding import funding_rate_for_row
from backtest_engine.accounting.futures import (
    FuturesInstrumentSpec,
    FuturesPortfolioSpec,
    apply_contract_risk,
)
from backtest_engine.core.exceptions import DatasetValidationError, StrategyPluginError
from backtest_engine.core.models import StrategyConfig
from backtest_engine.core.quality_approval import (
    explicit_usable_decision_day_pct,
    is_session_gap_approval,
)
from backtest_engine.core.risk_safety import (
    assert_end_position,
    assert_post_fill_limits,
    assert_solvent,
)
from backtest_engine.execution.models import OrderRequest
from backtest_engine.execution.simulator import ExecutionSimulatorV3, bracket_exit_orders
from backtest_engine.intraday.event.config import (
    IntradayEventBacktestConfig,
    IntradayEventDatasetConfig,
    IntradayEventExecutionConfig,
    IntradayEventOutputConfig,
    IntradayEventStrategyConfig,
)
from backtest_engine.intraday.event.quality import (
    append_quality_threshold_violations,
    runtime_decision_coverage,
)
from backtest_engine.intraday.validation import enforce_causal_bar_timestamps
from backtest_engine.portfolio.multi.config import (
    MultiInstrumentExecutionConfig,
    MultiInstrumentUniverseConfig,
)
from backtest_engine.portfolio.multi.universe_runtime import MultiUniverseRuntime
from backtest_engine.strategies.lookahead import enforce_vectorized_lookahead
from backtest_engine.workbench.models import WorkbenchRunRequest


def execution_bar(*, open_: float, high: float, low: float) -> dict[str, object]:
    return {
        "ts": datetime(2024, 1, 2, 10),
        "open": open_,
        "high": high,
        "low": low,
        "close": open_,
        "volume": 100.0,
    }


def test_ohlc_orders_use_gap_aware_prices() -> None:
    stop = ExecutionSimulatorV3()
    stop.submit(
        OrderRequest(
            order_id="stop",
            side="sell",
            quantity=1.0,
            order_type="stop",
            stop_price=95.0,
        )
    )
    limit = ExecutionSimulatorV3()
    limit.submit(
        OrderRequest(
            order_id="limit",
            side="buy",
            quantity=1.0,
            order_type="limit",
            limit_price=95.0,
        )
    )

    assert stop.process_bar(execution_bar(open_=90.0, high=92.0, low=89.0), bar_index=0)[
        0
    ].price == pytest.approx(90.0)
    assert limit.process_bar(execution_bar(open_=90.0, high=92.0, low=89.0), bar_index=0)[
        0
    ].price == pytest.approx(90.0)


def test_stop_limit_remains_triggered_on_later_bars() -> None:
    simulator = ExecutionSimulatorV3(intrabar_ambiguity_policy="conservative")
    simulator.submit(
        OrderRequest(
            order_id="stop-limit",
            side="buy",
            quantity=1.0,
            order_type="stop_limit",
            stop_price=105.0,
            limit_price=95.0,
            time_in_force="gtc",
        )
    )

    assert (
        simulator.process_bar(execution_bar(open_=100.0, high=106.0, low=99.0), bar_index=0) == []
    )
    fills = simulator.process_bar(execution_bar(open_=96.0, high=100.0, low=94.0), bar_index=1)

    assert simulator.orders["stop-limit"].stop_triggered is True
    assert fills[0].price == pytest.approx(95.0)


def test_market_impact_cannot_violate_limit_price() -> None:
    simulator = ExecutionSimulatorV3(market_impact_bps=100.0)
    simulator.submit(
        OrderRequest(
            order_id="limit-impact",
            side="buy",
            quantity=1.0,
            order_type="limit",
            limit_price=100.0,
        )
    )

    fills = simulator.process_bar(execution_bar(open_=99.5, high=101.0, low=99.0), bar_index=0)

    assert fills[0].price == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("policy", "expected_order"),
    [("conservative", "entry:stop_loss"), ("optimistic", "entry:take_profit")],
)
def test_ambiguous_oco_bar_uses_declared_policy(policy: str, expected_order: str) -> None:
    simulator = ExecutionSimulatorV3(intrabar_ambiguity_policy=policy)
    for request in bracket_exit_orders(
        parent_order_id="entry",
        side="sell",
        quantity=1.0,
        take_profit_price=110.0,
        stop_loss_price=95.0,
    ):
        simulator.submit(request)

    fills = simulator.process_bar(execution_bar(open_=100.0, high=111.0, low=94.0), bar_index=0)

    assert [fill.order_id for fill in fills] == [expected_order]


def test_noncausal_timing_and_legacy_multi_latency_fail_closed() -> None:
    with pytest.raises(ValidationError, match="same close"):
        IntradayEventExecutionConfig(timing="signal_close_to_same_close")
    with pytest.raises(ValidationError, match="non-causal"):
        MultiInstrumentExecutionConfig(timing="signal_close_to_same_close")
    with pytest.raises(ValidationError, match="engine=v3"):
        MultiInstrumentExecutionConfig(engine="legacy", latency_bars=1)


def test_legacy_bar_timestamps_require_explicit_diagnostic_override() -> None:
    validation = {"data_kind": "bars", "bar_time_semantics": "legacy_ts"}
    with pytest.raises(DatasetValidationError, match="OHLC lookahead"):
        enforce_causal_bar_timestamps(validation, allow_legacy=False, label="fixture")
    enforce_causal_bar_timestamps(validation, allow_legacy=True, label="fixture")


def test_held_leg_remains_in_accounting_when_session_blocks_decision() -> None:
    runtime = MultiUniverseRuntime(MultiInstrumentUniverseConfig(), ["A"])

    route = runtime.route(
        datetime(2024, 1, 2, 20),
        exact_assets=["A"],
        decision_assets=[],
        positions={"A": 1.0},
    )

    assert route.strategy_assets == []
    assert route.execution_assets == ["A"]


def test_pipeline_approval_uses_runtime_decision_coverage(tmp_path: Path) -> None:
    approval = {
        "approval_status": "approved",
        "criteria": {"max_gap_ratio": 0.005},
        "gap_audit": {"status": "warning", "session_aware": True, "gap_ratio": 0.002},
    }
    config = IntradayEventBacktestConfig(
        dataset=IntradayEventDatasetConfig(path=tmp_path / "bars.parquet"),
        strategy=IntradayEventStrategyConfig(strategy_file=tmp_path / "strategy.py"),
        execution=IntradayEventExecutionConfig(
            from_date=date(2024, 1, 2),
            schedule="times",
            decision_times=[time(14, 40)],
            decision_tolerance_minutes=0,
        ),
        output=IntradayEventOutputConfig(root=tmp_path, run_id="fixture"),
    )
    violations: list[str] = []
    append_quality_threshold_violations(violations, config, approval)
    frame = pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 2, 14, 40),
                datetime(2024, 1, 3, 14, 40),
                datetime(2024, 1, 3, 14, 45),
                datetime(2024, 1, 3, 14, 50),
            ]
        }
    )

    assert violations == []
    assert explicit_usable_decision_day_pct(approval) is None
    assert is_session_gap_approval(approval) is True
    above_threshold = {
        **approval,
        "gap_audit": {**approval["gap_audit"], "gap_ratio": 0.006},
    }
    assert is_session_gap_approval(above_threshold) is False
    assert runtime_decision_coverage(config, frame, 5) == (1.0, 2, 2)


def test_workbench_declares_safe_continuous_series_defaults() -> None:
    request = WorkbenchRunRequest(
        dataset_path=Path("bars.parquet"),
        script_source="function onDecision() { return 0; }",
    )

    assert request.end_of_data_position_policy == "mark"
    assert request.roll_pnl_policy == "exclude_gap"
    assert request.volume_participation_limit == pytest.approx(0.1)


def test_funding_is_charged_only_on_explicit_events() -> None:
    kwargs = {
        "rate_column": "funding_rate",
        "default_rate": 0.01,
        "event_column": "funding_event",
        "require_event_flag": True,
    }
    assert funding_rate_for_row({"funding_rate": 0.0}, **kwargs) == 0.0
    assert funding_rate_for_row({"funding_rate": 0.01, "funding_event": False}, **kwargs) == 0.0
    assert funding_rate_for_row(
        {"funding_rate": 0.01, "funding_event": True}, **kwargs
    ) == pytest.approx(0.01)
    with pytest.raises(DatasetValidationError, match="explicit funding_event=true"):
        funding_rate_for_row({"funding_rate": 0.01}, **kwargs)


def test_risk_invariants_reject_gap_breaches_and_nonpositive_equity() -> None:
    with pytest.raises(DatasetValidationError, match="max_position_notional"):
        assert_post_fill_limits(
            equity_cash=100.0,
            gross_notional=150.0,
            margin_requirement=10.0,
            max_position_notional=120.0,
        )
    with pytest.raises(DatasetValidationError, match="insolvent"):
        assert_solvent(0.0)
    with pytest.raises(DatasetValidationError, match="Open position"):
        assert_end_position(1.0, "reject", label="fixture")


def test_negative_equity_cannot_produce_reversed_clip_position() -> None:
    result = apply_contract_risk(
        contracts=2.0,
        fill_price=100.0,
        instrument=FuturesInstrumentSpec(contract_multiplier=10.0),
        portfolio=FuturesPortfolioSpec(
            initial_capital=100.0,
            max_leverage=1.0,
            risk_policy="clip",
        ),
        equity_cash=-1.0,
    )

    assert result.contracts == 0.0
    assert result.max_contracts == 0.0


def test_vectorized_plugin_lookahead_gate_is_mandatory(tmp_path: Path) -> None:
    strategy = tmp_path / "leaking.py"
    strategy.write_text(
        "def generate_signals(ctx):\n"
        "    values = ctx.series('close').to_list()\n"
        "    return [float(value < values[-1]) for value in values]\n",
        encoding="utf-8",
    )
    frame = pl.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]})

    with pytest.raises(StrategyPluginError, match="mandatory lookahead audit"):
        enforce_vectorized_lookahead(
            frame,
            StrategyConfig(name="plugin", strategy_file=strategy),
        )
