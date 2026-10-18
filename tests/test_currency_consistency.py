from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest
from test_intraday_event_dataset_support import (
    write_intraday_dataset as write_event_dataset,
)
from test_intraday_event_dataset_support import write_intraday_plugin

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.intraday.engine import validate_intraday_dataset
from backtest_engine.intraday.event.engine import (
    IntradayEventBacktestConfig,
    IntradayEventDatasetConfig,
    IntradayEventExecutionConfig,
    IntradayEventInstrumentConfig,
    IntradayEventInstrumentSpec,
    IntradayEventOutputConfig,
    IntradayEventPortfolioConfig,
    IntradayEventStrategyConfig,
    run_intraday_event_backtest,
)


@pytest.mark.parametrize("invalid_close", [float("nan"), float("inf"), 0.0, -1.0])
def test_intraday_validation_rejects_non_finite_or_non_positive_close(
    tmp_path,
    invalid_close: float,
) -> None:
    dataset_path = write_event_dataset(tmp_path)
    frame = pl.read_parquet(dataset_path).with_columns(
        pl.when(pl.int_range(pl.len()) == 1)
        .then(pl.lit(invalid_close))
        .otherwise(pl.col("close"))
        .alias("close")
    )

    with pytest.raises(DatasetValidationError, match="close must be finite and positive"):
        validate_intraday_dataset(frame, dataset_path)


def test_signal_units_cash_and_trade_pnl_stay_in_base_currency(tmp_path) -> None:
    dataset_path = write_event_dataset(tmp_path, closes=[100.0, 110.0])
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")
    result = run_intraday_event_backtest(
        IntradayEventBacktestConfig(
            dataset=IntradayEventDatasetConfig(
                path=dataset_path,
                quality_approval_required=False,
                allow_legacy_bar_timestamps=True,
            ),
            strategy=IntradayEventStrategyConfig(strategy_file=plugin_path),
            instrument=IntradayEventInstrumentConfig(
                price_currency="USD",
                pnl_currency="USD",
            ),
            portfolio=IntradayEventPortfolioConfig(
                initial_capital=1000.0,
                base_currency="RUB",
                fx_rates={"USD/RUB": 90.0},
                sizing_mode="signal_units",
            ),
            execution=IntradayEventExecutionConfig(
                timing="signal_close_to_same_close",
                allow_noncausal_same_close=True,
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
            ),
            output=IntradayEventOutputConfig(
                root=tmp_path / "out",
                run_id="signal_fx",
                write_csv=False,
            ),
        )
    )
    events = pl.read_parquet(result.output_dir / "events.parquet")
    trades = pl.read_parquet(result.output_dir / "trades.parquet")
    ledger = pl.read_parquet(result.output_dir / "portfolio_ledger.parquet")

    assert result.metrics["final_equity_cash"] == pytest.approx(1100.0)
    assert events.row(0, named=True)["pnl_currency"] == "RUB"
    assert ledger.row(0, named=True)["currency"] == "RUB"
    assert trades.get_column("gross_pnl").sum() == pytest.approx(
        events.get_column("gross_pnl").sum()
    )


def test_contract_cash_is_converted_once_into_portfolio_currency(tmp_path) -> None:
    dataset_path = write_event_dataset(tmp_path, closes=[100.0, 101.0])
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")
    result = run_intraday_event_backtest(
        IntradayEventBacktestConfig(
            dataset=IntradayEventDatasetConfig(
                path=dataset_path,
                quality_approval_required=False,
                allow_legacy_bar_timestamps=True,
            ),
            strategy=IntradayEventStrategyConfig(strategy_file=plugin_path),
            instrument=IntradayEventInstrumentConfig(
                price_currency="USD",
                pnl_currency="USD",
            ),
            portfolio=IntradayEventPortfolioConfig(
                initial_capital=1000.0,
                base_currency="RUB",
                fx_rates={"USD/RUB": 90.0},
                sizing_mode="fixed_contracts",
            ),
            execution=IntradayEventExecutionConfig(
                timing="signal_close_to_same_close",
                allow_noncausal_same_close=True,
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
            ),
            output=IntradayEventOutputConfig(
                root=tmp_path / "out",
                run_id="contract_fx",
                write_csv=False,
            ),
        )
    )
    event = pl.read_parquet(result.output_dir / "events.parquet").row(0, named=True)

    assert event["gross_pnl"] == pytest.approx(90.0)
    assert event["pnl_currency"] == "RUB"
    assert event["position_notional"] == pytest.approx(9000.0)
    assert event["point_value"] == pytest.approx(90.0)
    assert result.metrics["final_equity_cash"] == pytest.approx(1090.0)
    trade = pl.read_parquet(result.output_dir / "trades.parquet").row(0, named=True)
    assert trade["point_value"] == pytest.approx(90.0)
    assert trade["entry_notional"] == pytest.approx(9000.0)
    assert trade["exit_notional"] == pytest.approx(9090.0)
    assert trade["gross_return"] == pytest.approx(0.09)


def test_trade_attribution_reconciles_contract_resizing_costs(tmp_path) -> None:
    start = datetime(2024, 1, 2, 10, 0)
    dataset_path = write_event_dataset(
        tmp_path,
        timestamps=[start + timedelta(minutes=5 * index) for index in range(4)],
        closes=[100.0, 110.0, 120.0, 120.0],
    )
    plugin_path = write_intraday_plugin(
        tmp_path,
        """
def generate_signal(ctx):
    close = float(ctx.current_bar()["close"])
    if close < 110.0:
        return {"target": 1.0, "contracts": 2.0}
    if close < 120.0:
        return {"target": 1.0, "contracts": 1.0}
    return {"target": 0.0, "contracts": 0.0}
""".lstrip(),
    )
    result = run_intraday_event_backtest(
        IntradayEventBacktestConfig(
            dataset=IntradayEventDatasetConfig(
                path=dataset_path,
                quality_approval_required=False,
                allow_legacy_bar_timestamps=True,
            ),
            strategy=IntradayEventStrategyConfig(strategy_file=plugin_path),
            portfolio=IntradayEventPortfolioConfig(
                initial_capital=1000.0,
                sizing_mode="strategy_contracts",
            ),
            execution=IntradayEventExecutionConfig(
                timing="signal_close_to_same_close",
                allow_noncausal_same_close=True,
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
                commission_bps=10.0,
            ),
            output=IntradayEventOutputConfig(
                root=tmp_path / "out",
                run_id="contract_resize",
                write_csv=False,
            ),
        )
    )
    events = pl.read_parquet(result.output_dir / "events.parquet")
    trades = pl.read_parquet(result.output_dir / "trades.parquet")

    assert trades.height == 1
    assert trades.get_column("gross_pnl").sum() == pytest.approx(
        events.get_column("gross_pnl").sum()
    )
    assert trades.get_column("cost_cash").sum() == pytest.approx(
        events.get_column("cost_cash").sum()
    )
    assert trades.get_column("net_pnl").sum() == pytest.approx(events.get_column("net_pnl").sum())
    trade = trades.row(0, named=True)
    assert trade["entry_contracts"] == 2.0
    assert trade["exit_contracts"] == 1.0
    assert trade["entry_notional"] == pytest.approx(200.0)
    assert trade["exit_notional"] == pytest.approx(120.0)
    assert trade["gross_return"] == pytest.approx(0.03)


def test_notional_sizing_uses_decision_price_without_future_fill_lookahead(tmp_path) -> None:
    start = datetime(2024, 1, 2, 10, 0)
    dataset_path = write_event_dataset(
        tmp_path,
        timestamps=[start + timedelta(minutes=5 * index) for index in range(3)],
        closes=[100.0, 200.0, 200.0],
    )
    pl.read_parquet(dataset_path).with_columns(
        pl.Series("active_contract", ["OLD", "NEW", "NEW"]),
        pl.Series("source_contract_symbol", ["OLD", "NEW", "NEW"]),
    ).write_parquet(dataset_path)
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")
    result = run_intraday_event_backtest(
        IntradayEventBacktestConfig(
            dataset=IntradayEventDatasetConfig(
                path=dataset_path,
                quality_approval_required=False,
                allow_legacy_bar_timestamps=True,
            ),
            strategy=IntradayEventStrategyConfig(strategy_file=plugin_path),
            instrument=IntradayEventInstrumentConfig(
                specs=[
                    IntradayEventInstrumentSpec(symbol="OLD", contract_multiplier=1.0),
                    IntradayEventInstrumentSpec(symbol="NEW", contract_multiplier=10.0),
                ]
            ),
            portfolio=IntradayEventPortfolioConfig(
                initial_capital=10_000.0,
                sizing_mode="notional",
                target_notional=1000.0,
            ),
            execution=IntradayEventExecutionConfig(
                timing="signal_close_to_next_bar_close",
                end_of_data_position_policy="mark",
                from_date=date(2024, 1, 2),
            ),
            output=IntradayEventOutputConfig(
                root=tmp_path / "out",
                run_id="decision_sizing",
                write_csv=False,
            ),
        )
    )
    event = pl.read_parquet(result.output_dir / "events.parquet").row(0, named=True)

    assert event["requested_contracts"] == pytest.approx(10.0)
    assert event["fill_price"] == pytest.approx(200.0)
    assert event["position_notional"] == pytest.approx(20_000.0)
