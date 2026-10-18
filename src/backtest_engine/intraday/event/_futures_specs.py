from __future__ import annotations

from backtest_engine.accounting.futures import (
    FuturesExecutionCostSpec,
)
from backtest_engine.intraday.event.config import (
    IntradayEventExecutionConfig,
    IntradayEventInstrumentSpec,
    IntradayEventPortfolioConfig,
)


def event_cash_currency(
    instrument: IntradayEventInstrumentSpec,
    portfolio: IntradayEventPortfolioConfig,
) -> str:
    del instrument
    return portfolio.base_currency


def futures_execution_cost_spec(
    execution: IntradayEventExecutionConfig,
) -> FuturesExecutionCostSpec:
    return FuturesExecutionCostSpec(
        commission_bps=execution.commission_bps,
        slippage_bps=execution.slippage_bps,
        slippage_ticks=execution.slippage_ticks,
        bid_ask_spread_bps=execution.bid_ask_spread_bps,
        bid_ask_spread_ticks=execution.bid_ask_spread_ticks,
        roll_cost_bps=execution.roll_cost_bps,
        roll_slippage_ticks=execution.roll_slippage_ticks,
    )
