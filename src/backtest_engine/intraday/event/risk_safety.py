from __future__ import annotations

from typing import Any

from backtest_engine.core.risk_safety import (
    assert_end_position,
    assert_post_fill_limits,
    assert_solvent,
)
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig
from backtest_engine.intraday.event.crypto_rules import validate_crypto_fill


def validate_event_post_fill(
    config: IntradayEventBacktestConfig,
    instrument: Any,
    instrument_state: dict[str, Any],
    contracts: float,
    equity_cash: float,
    *,
    previous_contracts: float = 0.0,
    fill_price: float | None = None,
) -> None:
    if fill_price is not None:
        validate_crypto_fill(
            instrument, previous_contracts, contracts, fill_price, config.portfolio.leverage
        )
    assert_post_fill_limits(
        equity_cash=equity_cash,
        gross_notional=float(instrument_state["position_notional"]),
        margin_requirement=event_margin(config, instrument_state, contracts),
        max_position_notional=config.portfolio.max_position_notional,
        max_leverage=config.portfolio.max_leverage,
        max_margin_usage=config.portfolio.max_margin_usage,
        label=str(instrument.symbol),
    )


def validate_event_after_settlement(
    config: IntradayEventBacktestConfig,
    instrument: Any,
    instrument_state: dict[str, Any],
    contracts: float,
    equity_cash: float,
) -> None:
    assert_solvent(
        equity_cash,
        margin_requirement=event_margin(config, instrument_state, contracts),
        label=str(instrument.symbol),
    )


def validate_event_end(config: IntradayEventBacktestConfig, contracts: float) -> None:
    assert_end_position(
        contracts,
        config.execution.end_of_data_position_policy,
        label=str(config.dataset.path),
    )


def event_margin(
    config: IntradayEventBacktestConfig,
    instrument_state: dict[str, Any],
    contracts: float,
) -> float | None:
    if config.portfolio.sizing_mode == "signal_units":
        return None
    initial_margin = instrument_state["initial_margin"]
    return None if initial_margin is None else abs(contracts) * float(initial_margin)
