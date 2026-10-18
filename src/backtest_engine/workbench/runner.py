"""Builds and executes one Workbench backtest request with complete artifacts."""

from __future__ import annotations

import hashlib
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from backtest_engine.cli.parsing import (
    parse_date,
    parse_intraday_event_missing_decision_policy,
    parse_intraday_event_schedule,
    parse_intraday_event_timing,
)
from backtest_engine.intraday.event.engine import (
    IntradayEventBacktestConfig,
    IntradayEventDatasetConfig,
    IntradayEventExecutionConfig,
    IntradayEventInstrumentConfig,
    IntradayEventOutputConfig,
    IntradayEventPortfolioConfig,
    IntradayEventSessionConfig,
    IntradayEventStrategyConfig,
    run_intraday_event_backtest,
)
from backtest_engine.intraday.event.loop import EventProgressCallback
from backtest_engine.strategies.runtime import StrategyMetadata
from backtest_engine.workbench.history import (
    append_workbench_history,
    history_path_for,
    success_record,
)
from backtest_engine.workbench.market_defaults import account_defaults
from backtest_engine.workbench.models import WorkbenchRunRequest, WorkbenchRunResponse
from backtest_engine.workbench.report import build_workbench_report
from backtest_engine.workbench.sdk_catalog import freeze_catalog_source
from backtest_engine.workbench.strategy_contract import (
    load_workbench_strategy_metadata,
    strategy_decision_times,
)

MAX_SCRIPT_SOURCE_BYTES = 256 * 1024


def run_workbench_backtest(
    request: WorkbenchRunRequest,
    progress_callback: EventProgressCallback | None = None,
) -> WorkbenchRunResponse:
    run_id = safe_run_id(request.run_id or generated_run_id(request))
    script_path = materialize_script(request, run_id)
    metadata = load_workbench_strategy_metadata(script_path, request.params)
    report_phase(progress_callback, pct=0.08, stage="strategy", message="strategy validated")
    config = build_backtest_config(request, run_id, script_path, metadata)
    result = run_intraday_event_backtest(config, progress_callback=progress_callback)
    if script_path.suffix == ".py" and script_path.read_text(encoding="utf-8").startswith(
        "SDK_CATALOG_SNAPSHOT = "
    ):
        shutil.copyfile(script_path, result.output_dir / "strategy_snapshot.py")
    report_phase(progress_callback, pct=0.88, stage="report", message="building report")
    report = build_workbench_report(
        result.output_dir,
        dataset_path=request.dataset_path,
        max_points=request.chart_limit,
    )
    append_workbench_history(
        history_path_for(request.output_root),
        success_record(
            request.model_copy(
                update={
                    "decision_times": [
                        value.strftime("%H:%M") for value in config.execution.decision_times
                    ]
                }
            ),
            run_id=run_id,
            output_dir=result.output_dir,
            report=report,
        ),
    )
    report_phase(progress_callback, pct=0.97, stage="report", message="report ready")
    return WorkbenchRunResponse(
        status="completed",
        run_id=run_id,
        output_dir=result.output_dir,
        report=report,
    )


def report_phase(
    progress_callback: EventProgressCallback | None,
    *,
    pct: float,
    stage: str,
    message: str,
) -> None:
    phase = getattr(progress_callback, "phase", None)
    if callable(phase):
        phase(pct=pct, stage=stage, message=message)


def build_backtest_config(
    request: WorkbenchRunRequest,
    run_id: str,
    script_path: Path,
    metadata: StrategyMetadata,
) -> IntradayEventBacktestConfig:
    defaults = account_defaults(request)
    return IntradayEventBacktestConfig(
        dataset=IntradayEventDatasetConfig(
            path=request.dataset_path,
            quality_approval_path=request.quality_approval_path,
            quality_approval_required=request.quality_approval_required,
            allow_legacy_bar_timestamps=request.allow_legacy_bar_timestamps,
            catalog_path=request.catalog_path,
            dataset_id=request.dataset_id,
            catalog_enforcement=request.catalog_enforcement,
        ),
        strategy=IntradayEventStrategyConfig(
            strategy_file=script_path,
            strategy_params=dict(request.params),
            strict_runtime=request.strict_runtime,
            timeframe=metadata.timeframe,
        ),
        instrument=IntradayEventInstrumentConfig(
            contract_multiplier=request.contract_multiplier,
            lot_size=request.lot_size,
            tick_size=request.tick_size,
            tick_value=request.tick_value,
            exchange_fee_per_contract=request.exchange_fee_per_contract,
            registry_path=request.instrument_registry_path,
        ),
        portfolio=IntradayEventPortfolioConfig(
            initial_capital=request.initial_capital,
            sizing_mode=request.sizing_mode,
            fixed_contracts=request.fixed_contracts,
            target_notional=request.target_notional,
            contract_rounding=request.contract_rounding,
            allow_fractional_contracts=defaults["allow_fractional_contracts"],
            base_currency=defaults["base_currency"],
            fx_rates=request.fx_rates,
            valuation_rates_path=request.valuation_rates_path,
            collateral_currency=request.collateral_currency,
            initial_collateral=request.initial_collateral,
            margin_mode=request.margin_mode,
            leverage=request.leverage,
            require_mark_price=defaults["require_mark_price"],
            require_margin_tiers=defaults["require_margin_tiers"],
            max_position_notional=request.max_position_notional,
            max_leverage=request.max_leverage,
            max_margin_usage=request.max_margin_usage,
            risk_policy=request.risk_policy,
        ),
        session=IntradayEventSessionConfig(
            calendar_path=request.calendar_path,
            session_filter=request.session_filter,
        ),
        execution=IntradayEventExecutionConfig(
            engine=request.execution_engine,
            timing=parse_intraday_event_timing(request.execution_timing),
            allow_noncausal_same_close=request.allow_noncausal_same_close,
            end_of_data_position_policy=request.end_of_data_position_policy,
            from_date=parse_date(request.from_date),
            schedule=parse_intraday_event_schedule(metadata.schedule),
            decision_times=strategy_decision_times(metadata, request.params),
            decision_tolerance_minutes=metadata.decision_tolerance_minutes,
            missing_decision_policy=parse_intraday_event_missing_decision_policy(
                metadata.missing_decision_policy
            ),
            commission_bps=request.commission_bps,
            slippage_bps=request.slippage_bps,
            slippage_ticks=request.slippage_ticks,
            bid_ask_spread_bps=request.bid_ask_spread_bps,
            bid_ask_spread_ticks=request.bid_ask_spread_ticks,
            volume_participation_limit=request.volume_participation_limit,
            volume_to_contracts_multiplier=request.volume_to_contracts_multiplier,
            latency_bars=request.latency_bars,
            queue_ahead_contracts=request.queue_ahead_contracts,
            market_impact_bps=request.market_impact_bps,
            partial_fill_policy=request.partial_fill_policy,
            intrabar_ambiguity_policy=request.intrabar_ambiguity_policy,
            roll_cost_bps=request.roll_cost_bps,
            roll_slippage_ticks=request.roll_slippage_ticks,
            roll_pnl_policy=request.roll_pnl_policy,
            trading_days_per_year=defaults["trading_days_per_year"],
            liquidity_role=request.liquidity_role,
            funding_mode=request.funding_mode,
            funding_rate_column=request.funding_rate_column,
            default_funding_rate=request.default_funding_rate,
            funding_event_column=request.funding_event_column,
            require_funding_event_flag=request.require_funding_event_flag,
        ),
        output=IntradayEventOutputConfig(
            root=request.output_root,
            run_id=run_id,
            write_csv=False,
        ),
    )


def materialize_script(request: WorkbenchRunRequest, run_id: str) -> Path:
    if request.script_path is not None:
        language = "python" if request.script_path.suffix.lower() == ".py" else "javascript"
        original = request.script_path.read_text(encoding="utf-8-sig")
        source = freeze_catalog_source(original, language)
        if source == original:
            return request.script_path
    else:
        language = request.script_language
        source = freeze_catalog_source(request.script_source or "", language)
    if len(source.encode("utf-8")) > MAX_SCRIPT_SOURCE_BYTES:
        raise ValueError("script_source is too large for workbench execution")
    script_dir = request.output_root.parent / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".py" if language == "python" else ".js"
    script_path = script_dir / f"{run_id}{suffix}"
    script_path.write_text(source, encoding="utf-8")
    return script_path


def generated_run_id(request: WorkbenchRunRequest) -> str:
    digest = hashlib.sha256()
    digest.update(str(request.dataset_path).encode("utf-8"))
    digest.update(str(request.script_path).encode("utf-8"))
    digest.update((request.script_source or "").encode("utf-8"))
    digest.update(request.script_language.encode("utf-8"))
    digest.update(repr(sorted(request.params.items())).encode("utf-8"))
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"wb_{stamp}_{digest.hexdigest()[:8]}"


def safe_run_id(raw_value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_value.strip())
    value = value.strip("._-")
    if not value:
        raise ValueError("run_id is empty after sanitization")
    return value[:120]
