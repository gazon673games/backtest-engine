"""Stable public API for the domain-organized backtest engine package."""

import sys
from importlib import import_module

from backtest_engine.accounting.futures import (
    FuturesCostBreakdown,
    FuturesExecutionCostSpec,
    FuturesInstrumentSpec,
    FuturesPortfolioSpec,
    FuturesRiskResult,
    apply_contract_risk,
    contract_point_value,
    cost_breakdown,
    effective_tick_value,
    pnl_cash,
    position_notional,
    target_contracts,
    turnover_notional,
)
from backtest_engine.core.models import (
    BacktestConfig,
    BacktestResult,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
    PerformanceMetrics,
    QualityGateConfig,
    StrategyConfig,
)
from backtest_engine.daily.batch import BatchConfig, BatchResult, run_batch
from backtest_engine.daily.engine import inspect_dataset, run_backtest
from backtest_engine.data.lifecycle import (
    DataLifecycleAuditConfig,
    DataLifecycleAuditResult,
    audit_data_lifecycle,
)
from backtest_engine.intraday.batch.runner import (
    IntradayBatchConfig,
    IntradayBatchResult,
    IntradayBatchRunResult,
    run_intraday_batch,
)
from backtest_engine.intraday.engine import (
    IntradayBacktestConfig,
    IntradayBacktestResult,
    IntradayDatasetConfig,
    IntradayExecutionConfig,
    IntradayOutputConfig,
    IntradayStrategyConfig,
    run_intraday_backtest,
)
from backtest_engine.intraday.event.batch.runner import (
    IntradayEventBatchConfig,
    IntradayEventBatchResult,
    run_intraday_event_batch,
)
from backtest_engine.intraday.event.engine import (
    IntradayEventBacktestConfig,
    IntradayEventBacktestResult,
    IntradayEventDatasetConfig,
    IntradayEventExecutionConfig,
    IntradayEventOutputConfig,
    IntradayEventStrategyConfig,
    run_intraday_event_backtest,
)
from backtest_engine.intraday.event.reporting.report import (
    IntradayEventReportConfig,
    IntradayEventReportResult,
    generate_intraday_event_report,
)
from backtest_engine.portfolio.multi.engine import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentBacktestResult,
    MultiInstrumentExecutionConfig,
    MultiInstrumentLegConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
    MultiInstrumentStrategyContext,
    run_multi_instrument_backtest,
)
from backtest_engine.reporting.experiment import (
    ReportConfig,
    ReportResult,
    generate_experiment_report,
)
from backtest_engine.reporting.tearsheet.report import (
    IntradayEventTearsheetConfig,
    IntradayEventTearsheetResult,
    generate_intraday_event_tearsheet,
)
from backtest_engine.research.experiments.registry import (
    ExperimentCompareConfig,
    ExperimentCompareResult,
    ExperimentRegistryConfig,
    ExperimentRegistryResult,
    build_experiment_registry,
    generate_experiment_compare_report,
    load_experiment_registry,
)
from backtest_engine.research.pack import (
    ResearchPackConfig,
    ResearchPackResult,
    generate_research_pack,
)
from backtest_engine.strategies.api import StrategyContext
from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.sdk import StrategyParams

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BatchConfig",
    "BatchResult",
    "DataLifecycleAuditConfig",
    "DataLifecycleAuditResult",
    "DatasetConfig",
    "ExecutionConfig",
    "ExperimentCompareConfig",
    "ExperimentCompareResult",
    "ExperimentRegistryConfig",
    "ExperimentRegistryResult",
    "FuturesCostBreakdown",
    "FuturesExecutionCostSpec",
    "FuturesInstrumentSpec",
    "FuturesPortfolioSpec",
    "FuturesRiskResult",
    "IntradayBacktestConfig",
    "IntradayBacktestResult",
    "IntradayBatchConfig",
    "IntradayBatchResult",
    "IntradayBatchRunResult",
    "IntradayDatasetConfig",
    "IntradayEventBacktestConfig",
    "IntradayEventBacktestResult",
    "IntradayEventBatchConfig",
    "IntradayEventBatchResult",
    "IntradayEventDatasetConfig",
    "IntradayEventExecutionConfig",
    "IntradayEventOutputConfig",
    "IntradayEventReportConfig",
    "IntradayEventReportResult",
    "IntradayEventStrategyConfig",
    "IntradayEventTearsheetConfig",
    "IntradayEventTearsheetResult",
    "IntradayExecutionConfig",
    "IntradayOutputConfig",
    "IntradayStrategyConfig",
    "IntradayStrategyContext",
    "MultiInstrumentBacktestConfig",
    "MultiInstrumentBacktestResult",
    "MultiInstrumentExecutionConfig",
    "MultiInstrumentLegConfig",
    "MultiInstrumentOutputConfig",
    "MultiInstrumentPortfolioConfig",
    "MultiInstrumentStrategyConfig",
    "MultiInstrumentStrategyContext",
    "OutputConfig",
    "PerformanceMetrics",
    "QualityGateConfig",
    "ReportConfig",
    "ReportResult",
    "ResearchPackConfig",
    "ResearchPackResult",
    "StrategyConfig",
    "StrategyContext",
    "StrategyParams",
    "apply_contract_risk",
    "audit_data_lifecycle",
    "build_experiment_registry",
    "contract_point_value",
    "cost_breakdown",
    "effective_tick_value",
    "generate_experiment_compare_report",
    "generate_intraday_event_report",
    "generate_intraday_event_tearsheet",
    "generate_experiment_report",
    "generate_research_pack",
    "inspect_dataset",
    "load_experiment_registry",
    "pnl_cash",
    "position_notional",
    "run_batch",
    "run_backtest",
    "run_intraday_backtest",
    "run_intraday_batch",
    "run_intraday_event_backtest",
    "run_intraday_event_batch",
    "run_multi_instrument_backtest",
    "target_contracts",
    "turnover_notional",
]


# Keep established plugin imports working while implementation modules live in
# domain packages. New code should import from the paths on the right-hand side.
_LEGACY_MODULE_ALIASES = {
    "adapter_models": "backtest_engine.adapters.models",
    "adapter_registry": "backtest_engine.adapters.registry",
    "config": "backtest_engine.core.config",
    "exceptions": "backtest_engine.core.exceptions",
    "execution_v3": "backtest_engine.execution.simulator",
    "execution_v3_models": "backtest_engine.execution.models",
    "experiment_registry": "backtest_engine.research.experiments.registry",
    "futures_accounting": "backtest_engine.accounting.futures",
    "intraday_batch": "backtest_engine.intraday.batch.runner",
    "intraday_event": "backtest_engine.intraday.event.engine",
    "intraday_event_batch": "backtest_engine.intraday.event.batch.runner",
    "intraday_event_reporting": "backtest_engine.intraday.event.reporting.report",
    "intraday_strategy_api": "backtest_engine.strategies.intraday_api",
    "market_adapter": "backtest_engine.adapters.protocol",
    "moex_market_adapter": "backtest_engine.adapters.legacy_moex",
    "models": "backtest_engine.core.models",
    "multi_instrument": "backtest_engine.portfolio.multi.engine",
    "plugin_loader": "backtest_engine.strategies.plugins.loader",
    "quality": "backtest_engine.data.quality",
    "research_pack": "backtest_engine.research.pack",
    "research_validation": "backtest_engine.research.validation.gate",
    "strategy_api": "backtest_engine.strategies.api",
    "strategy_plugin_contract": "backtest_engine.strategies.contract",
    "strategy_runtime": "backtest_engine.strategies.runtime",
    "strategy_sdk": "backtest_engine.strategies.sdk",
    "tearsheet": "backtest_engine.reporting.tearsheet.report",
    "timeframes": "backtest_engine.core.timeframes",
}

for _legacy_name, _module_name in _LEGACY_MODULE_ALIASES.items():
    _module = import_module(_module_name)
    sys.modules[f"{__name__}.{_legacy_name}"] = _module
    setattr(sys.modules[__name__], _legacy_name, _module)
