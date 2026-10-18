# ruff: noqa
from test_multi_instrument_support import *  # noqa: F401,F403

from backtest_engine.cli.parsing import parse_multi_instrument_leg
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.portfolio.multi.config import MultiInstrumentInstrumentSpec
from backtest_engine.portfolio.multi.event_rows import build_leg_rows_for_time
from backtest_engine.portfolio.multi.interval_accounting import calculate_multi_interval
from backtest_engine.portfolio.multi.timeline import validate_execution_timeline


def _bars(asset_id: str, timestamps: list[datetime], prices: list[float], volume: float = 1000.0):
    return [
        {
            "ts": timestamp,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume,
            "active_contract": asset_id,
            "currency": "RUB",
            "source_contract_symbol": asset_id,
        }
        for timestamp, price in zip(timestamps, prices, strict=True)
    ]


def _config(legs, **execution):
    return MultiInstrumentBacktestConfig(
        legs=legs,
        strategy=MultiInstrumentStrategyConfig(strategy_file=Path("strategy.py")),
        portfolio=MultiInstrumentPortfolioConfig(base_currency="RUB", initial_capital=1000.0),
        execution=MultiInstrumentExecutionConfig(
            timing="signal_close_to_same_close",
            allow_noncausal_same_close=True,
            end_of_data_position_policy="mark",
            from_date=date(2024, 1, 2),
            **execution,
        ),
        output=MultiInstrumentOutputConfig(run_id="audit"),
    )


def test_multi_quality_approval_is_required_by_default() -> None:
    leg = MultiInstrumentLegConfig(asset_id="A", dataset_path=Path("A.parquet"))

    assert leg.quality_approval_required is True
    assert parse_multi_instrument_leg("A=A.parquet").quality_approval_required is True


def test_all_or_none_rejection_preserves_existing_position_pnl() -> None:
    start = datetime(2024, 1, 2, 10, 0)
    end = datetime(2024, 1, 2, 10, 5)
    legs = [
        MultiInstrumentLegConfig(
            asset_id=asset_id,
            dataset_path=Path(f"{asset_id}.parquet"),
            quality_approval_required=False,
            allow_legacy_bar_timestamps=True,
        )
        for asset_id in ("A", "B")
    ]
    rows_by_asset = {
        "A": _bars("A", [start, end], [100.0, 110.0]),
        "B": _bars("B", [start, end], [100.0, 100.0], volume=1.0),
    }
    contracts = {"A": 1.0, "B": 0.0}
    targets = {"A": 1.0, "B": 0.0}
    rows = build_leg_rows_for_time(
        current_ts=start,
        tradable_assets=["A", "B"],
        exact_indices={"A": {start: 0}, "B": {start: 0}},
        rows_by_asset=rows_by_asset,
        loaded_legs={leg.asset_id: {"config": leg} for leg in legs},
        config=_config(
            legs,
            rebalance_policy="all_or_none",
            volume_participation_limit=0.1,
        ),
        previous_targets_snapshot=dict(targets),
        current_targets=targets,
        current_contracts=contracts,
        target_targets={"A": 1.0, "B": 1.0},
        target_contracts={"A": 1.0, "B": 10.0},
        reasons={},
        features={},
        risk_result={"status": "approved", "scale": 1.0},
        previous_equity_cash=1000.0,
    )

    first = rows[0]
    assert first["order_status"] == "rebalance_rejected"
    assert first["target_contracts"] == 1.0
    assert first["gross_pnl"] == pytest.approx(10.0)
    assert first["net_pnl"] == pytest.approx(10.0)


def test_multi_roll_requires_policy_and_charges_both_standard_legs() -> None:
    fill = {"active_contract": "OLD"}
    exit_row = {"active_contract": "NEW"}
    instrument = MultiInstrumentInstrumentSpec()

    with pytest.raises(DatasetValidationError, match="roll_pnl_policy=exclude_gap"):
        calculate_multi_interval(
            fill_row=fill,
            exit_row=exit_row,
            target_contracts=1.0,
            contract_delta=0.0,
            fill_price=100.0,
            exit_price=200.0,
            fill_instrument=instrument,
            exit_instrument=instrument,
            execution=MultiInstrumentExecutionConfig(),
        )

    result = calculate_multi_interval(
        fill_row=fill,
        exit_row=exit_row,
        target_contracts=1.0,
        contract_delta=0.0,
        fill_price=100.0,
        exit_price=200.0,
        fill_instrument=instrument,
        exit_instrument=instrument,
        execution=MultiInstrumentExecutionConfig(
            commission_bps=100.0,
            roll_pnl_policy="exclude_gap",
        ),
    )

    assert result.gross_pnl == 0.0
    assert result.roll_pnl_excluded is True
    assert result.roll_turnover_notional == pytest.approx(300.0)
    assert result.costs.commission_cost_cash == pytest.approx(3.0)


def test_multi_timeline_rejects_position_visibility_before_fill() -> None:
    a_times = [
        datetime(2024, 1, 2, 10, 0),
        datetime(2024, 1, 2, 10, 5),
        datetime(2024, 1, 2, 10, 10),
    ]
    b_times = [
        datetime(2024, 1, 2, 10, 2),
        datetime(2024, 1, 2, 10, 7),
        datetime(2024, 1, 2, 10, 12),
    ]
    rows = {
        "A": _bars("A", a_times, [100.0, 101.0, 102.0]),
        "B": _bars("B", b_times, [100.0, 101.0, 102.0]),
    }
    exact = {"A": {a_times[0]: 0}, "B": {b_times[0]: 0}}

    with pytest.raises(DatasetValidationError, match="before fill_ts"):
        validate_execution_timeline(exact, rows, "signal_close_to_next_bar_close")
