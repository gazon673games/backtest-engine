"""Composes synchronized multi-instrument strategy runs."""

from __future__ import annotations

from backtest_engine.data.market_identity import account_currency, annualization_days
from backtest_engine.execution.calibration import calibration_updates
from backtest_engine.portfolio.multi.config import (
    MULTI_INSTRUMENT_CATALOG_ENFORCEMENTS,
    MULTI_INSTRUMENT_RISK_POLICIES,
    MULTI_INSTRUMENT_SESSION_FILTERS,
    MULTI_INSTRUMENT_SHORT_DAY_POLICIES,
    MULTI_INSTRUMENT_SIZING_MODES,
    MultiInstrumentBacktestConfig,
    MultiInstrumentBacktestResult,
    MultiInstrumentExecutionConfig,
    MultiInstrumentInstrumentSpec,
    MultiInstrumentLegConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
    MultiInstrumentUniverseConfig,
)
from backtest_engine.portfolio.multi.context import MultiInstrumentStrategyContext
from backtest_engine.portfolio.multi.data import load_legs
from backtest_engine.portfolio.multi.events import build_multi_instrument_events
from backtest_engine.portfolio.multi.outputs import save_multi_instrument_outputs
from backtest_engine.portfolio.multi.plugin import load_multi_instrument_strategy_plugin
from backtest_engine.portfolio.multi.sdk import SDKPortfolioStrategy

__all__ = [
    "MULTI_INSTRUMENT_CATALOG_ENFORCEMENTS",
    "MULTI_INSTRUMENT_RISK_POLICIES",
    "MULTI_INSTRUMENT_SESSION_FILTERS",
    "MULTI_INSTRUMENT_SHORT_DAY_POLICIES",
    "MULTI_INSTRUMENT_SIZING_MODES",
    "MultiInstrumentBacktestConfig",
    "MultiInstrumentBacktestResult",
    "MultiInstrumentExecutionConfig",
    "MultiInstrumentInstrumentSpec",
    "MultiInstrumentLegConfig",
    "MultiInstrumentOutputConfig",
    "MultiInstrumentPortfolioConfig",
    "MultiInstrumentStrategyConfig",
    "MultiInstrumentUniverseConfig",
    "MultiInstrumentStrategyContext",
    "build_multi_instrument_events",
    "run_multi_instrument_backtest",
    "save_multi_instrument_outputs",
]


def run_multi_instrument_backtest(
    config: MultiInstrumentBacktestConfig,
) -> MultiInstrumentBacktestResult:
    loaded = load_legs(config.legs, config.execution.from_date, config.execution.till_date)
    legs = [payload["config"] for payload in loaded.values()]
    currency = account_currency(
        config.portfolio.base_currency,
        [leg.pnl_currency for leg in legs]
        + [spec.pnl_currency or leg.pnl_currency for leg in legs for spec in leg.specs],
    )
    config = config.model_copy(
        update={
            "legs": legs,
            "portfolio": config.portfolio.model_copy(update={"base_currency": currency}),
            "execution": config.execution.model_copy(
                update={
                    "trading_days_per_year": annualization_days(
                        config.execution.trading_days_per_year, [leg.calendar_path for leg in legs]
                    ),
                }
            ),
        }
    )
    if config.execution.calibration_path is not None:
        bar_minutes = min(int(payload["validation"]["bar_minutes"]) for payload in loaded.values())
        execution = config.execution.model_copy(
            update=calibration_updates(config.execution.calibration_path, bar_minutes)
        )
        config = config.model_copy(update={"execution": execution})
    generate_targets = (
        SDKPortfolioStrategy(config, loaded)
        if config.strategy.sdk_sleeves
        else load_multi_instrument_strategy_plugin(
            config.strategy.strategy_file,
            require_metadata=config.strategy.strict_runtime,
        )
    )
    order_history = []
    portfolio_legs, portfolio_equity, portfolio_ledger, metrics = build_multi_instrument_events(
        config,
        loaded,
        generate_targets,
        order_history,
    )
    output_dir = save_multi_instrument_outputs(
        config,
        loaded,
        portfolio_legs,
        portfolio_equity,
        portfolio_ledger,
        metrics,
        order_history,
        generate_targets if config.strategy.sdk_sleeves else None,
    )
    return MultiInstrumentBacktestResult(
        output_dir=output_dir,
        metrics=metrics,
        dataset_validations={asset_id: leg["validation"] for asset_id, leg in loaded.items()},
    )
