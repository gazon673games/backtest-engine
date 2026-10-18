# ruff: noqa
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from backtest_engine.cli import app
from backtest_engine.core.exceptions import DatasetValidationError, StrategyPluginError
from backtest_engine.intraday.engine import validate_intraday_dataset
from backtest_engine.intraday.event.engine import (
    IntradayEventBacktestConfig,
    IntradayEventDatasetConfig,
    IntradayEventExecutionConfig,
    IntradayEventInstrumentConfig,
    IntradayEventInstrumentSpec,
    IntradayEventOutputConfig,
    IntradayEventPortfolioConfig,
    IntradayEventSessionConfig,
    IntradayEventStrategyConfig,
    build_intraday_event_events,
    run_intraday_event_backtest,
)

from test_intraday_event_dataset_support import *  # noqa: F401,F403

runner = CliRunner()


def build_events(
    tmp_path: Path,
    dataset_path: Path,
    plugin_path: Path,
    **config_options: object,
) -> pl.DataFrame:
    df = pl.read_parquet(dataset_path)
    validation = validate_intraday_dataset(df, dataset_path)
    return build_intraday_event_events(
        make_config(tmp_path, dataset_path, plugin_path, **config_options),
        df,
        validation,
    )


def make_config(
    tmp_path: Path,
    dataset_path: Path,
    plugin_path: Path,
    *,
    run_id: str = "event",
    schedule: str = "every-bar",
    decision_times: list[str] | None = None,
    decision_tolerance_minutes: int | None = None,
    missing_decision_policy: str = "carry",
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    slippage_ticks: float = 0.0,
    bid_ask_spread_bps: float = 0.0,
    bid_ask_spread_ticks: float = 0.0,
    volume_participation_limit: float | None = None,
    volume_to_contracts_multiplier: float = 1.0,
    partial_fill_policy: str = "allow",
    intrabar_ambiguity_policy: str = "conservative",
    roll_cost_bps: float = 0.0,
    roll_slippage_ticks: float = 0.0,
    roll_pnl_policy: str = "reject",
    execution_timing: str = "signal_close_to_same_close",
    quality_approval_path: Path | None = None,
    quality_approval_required: bool = False,
    min_usable_decision_day_pct: float = 0.95,
    contract_multiplier: float = 1.0,
    lot_size: float = 1.0,
    tick_size: float | None = None,
    tick_value: float | None = None,
    initial_margin: float | None = None,
    instrument_specs: list[IntradayEventInstrumentSpec] | None = None,
    initial_capital: float = 1.0,
    sizing_mode: str = "signal_units",
    fixed_contracts: float = 1.0,
    target_notional: float | None = None,
    allow_fractional_contracts: bool = True,
    max_position_notional: float | None = None,
    max_leverage: float | None = None,
    max_margin_usage: float | None = None,
    risk_policy: str = "reject",
    session_filter: str = "all",
    holiday_dates: list[date] | None = None,
    short_day_dates: list[date] | None = None,
    short_day_policy: str = "allow",
    write_csv: bool = False,
) -> IntradayEventBacktestConfig:
    return IntradayEventBacktestConfig(
        dataset=IntradayEventDatasetConfig(
            path=dataset_path,
            quality_approval_path=quality_approval_path,
            quality_approval_required=quality_approval_required,
            min_usable_decision_day_pct=min_usable_decision_day_pct,
            allow_legacy_bar_timestamps=True,
        ),
        strategy=IntradayEventStrategyConfig(strategy_file=plugin_path),
        instrument=IntradayEventInstrumentConfig(
            price_currency="RUB",
            pnl_currency="RUB",
            contract_multiplier=contract_multiplier,
            lot_size=lot_size,
            tick_size=tick_size,
            tick_value=tick_value,
            initial_margin=initial_margin,
            specs=instrument_specs or [],
        ),
        portfolio=IntradayEventPortfolioConfig(
            base_currency="RUB",
            initial_capital=initial_capital,
            sizing_mode=sizing_mode,
            fixed_contracts=fixed_contracts,
            target_notional=target_notional,
            allow_fractional_contracts=allow_fractional_contracts,
            max_position_notional=max_position_notional,
            max_leverage=max_leverage,
            max_margin_usage=max_margin_usage,
            risk_policy=risk_policy,
        ),
        session=IntradayEventSessionConfig(
            main_session_start="10:00",
            main_session_end="18:50",
            evening_session_start="19:00",
            evening_session_end="23:50",
            session_filter=session_filter,
            holiday_dates=holiday_dates or [],
            short_day_dates=short_day_dates or [],
            short_day_policy=short_day_policy,
        ),
        execution=IntradayEventExecutionConfig(
            trading_days_per_year=252,
            timing=execution_timing,
            allow_noncausal_same_close=True,
            end_of_data_position_policy="mark",
            from_date=date(2024, 1, 2),
            schedule=schedule,
            decision_times=[
                datetime.strptime(value, "%H:%M").time() for value in decision_times or []
            ],
            decision_tolerance_minutes=decision_tolerance_minutes,
            missing_decision_policy=missing_decision_policy,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            slippage_ticks=slippage_ticks,
            bid_ask_spread_bps=bid_ask_spread_bps,
            bid_ask_spread_ticks=bid_ask_spread_ticks,
            volume_participation_limit=volume_participation_limit,
            volume_to_contracts_multiplier=volume_to_contracts_multiplier,
            partial_fill_policy=partial_fill_policy,
            intrabar_ambiguity_policy=intrabar_ambiguity_policy,
            roll_cost_bps=roll_cost_bps,
            roll_slippage_ticks=roll_slippage_ticks,
            roll_pnl_policy=roll_pnl_policy,
        ),
        output=IntradayEventOutputConfig(root=tmp_path / "out", run_id=run_id, write_csv=write_csv),
    )
