from __future__ import annotations

from typing import Any

from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime
from backtest_engine.core.risk_safety import assert_post_fill_limits, assert_solvent
from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig


def validate_post_fill_rows(
    rows: list[dict[str, Any]],
    config: MultiInstrumentBacktestConfig,
    ledger: BacktestLedgerRuntime,
    equity_cash: float,
) -> float:
    gross_notional = 0.0
    margin = 0.0
    for row in rows:
        currency = str(row["pnl_currency"])
        notional_base = ledger.amount_in_base(float(row["position_notional"]), currency)
        margin_base = ledger.amount_in_base(float(row["margin_requirement"]), currency)
        assert_post_fill_limits(
            equity_cash=equity_cash,
            gross_notional=notional_base,
            margin_requirement=None,
            max_position_notional=config.portfolio.max_position_notional,
            label=str(row["asset_id"]),
        )
        gross_notional += notional_base
        margin += margin_base
    assert_post_fill_limits(
        equity_cash=equity_cash,
        gross_notional=gross_notional,
        margin_requirement=margin,
        max_leverage=config.portfolio.max_leverage,
        max_margin_usage=config.portfolio.max_margin_usage,
        label="multi-instrument portfolio",
    )
    return margin


def validate_after_settlement(equity_cash: float, margin: float) -> None:
    assert_solvent(
        equity_cash,
        margin_requirement=margin,
        label="multi-instrument portfolio",
    )
