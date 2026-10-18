from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from backtest_engine.cli.common import CLI_EXCEPTIONS, raise_cli_error
from backtest_engine.cli.parsing import (
    parse_date,
    parse_intraday_event_missing_decision_policy,
    parse_intraday_event_schedule,
    parse_intraday_event_timing,
    parse_params,
    parse_time,
)
from backtest_engine.data.catalog import resolve_data_catalog_entry
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


def register(app: typer.Typer) -> None:
    @app.command(name="run-intraday-event-script")
    def run_intraday_event_script(
        dataset_path: Annotated[Path | None, typer.Option()] = None,
        catalog_path: Annotated[Path | None, typer.Option()] = None,
        dataset_id: Annotated[str | None, typer.Option()] = None,
        catalog_enforcement: Annotated[str, typer.Option()] = "off",
        strategy_file: Annotated[Path | None, typer.Option()] = None,
        param: Annotated[list[str] | None, typer.Option("--param")] = None,
        strict_runtime: Annotated[bool, typer.Option()] = False,
        quality_approval_path: Annotated[Path | None, typer.Option()] = None,
        require_quality_approval: Annotated[bool, typer.Option()] = True,
        min_usable_decision_day_pct: Annotated[float, typer.Option()] = 0.95,
        allow_legacy_bar_timestamps: Annotated[bool, typer.Option()] = False,
        from_date: Annotated[str, typer.Option()] = "0001-01-01",
        execution_timing: Annotated[str, typer.Option()] = "signal_close_to_next_bar_close",
        allow_noncausal_same_close: Annotated[bool, typer.Option()] = False,
        end_of_data_position_policy: Annotated[str, typer.Option()] = "reject",
        schedule: Annotated[str, typer.Option()] = "every-bar",
        decision_time: Annotated[list[str] | None, typer.Option("--decision-time")] = None,
        decision_tolerance_minutes: Annotated[int | None, typer.Option(min=0)] = None,
        missing_decision_policy: Annotated[str, typer.Option()] = "carry",
        sizing_mode: Annotated[str, typer.Option()] = "signal_units",
        initial_capital: Annotated[float, typer.Option(min=0.0)] = 1.0,
        fixed_contracts: Annotated[float, typer.Option(min=0.0)] = 1.0,
        target_notional: Annotated[float | None, typer.Option(min=0.0)] = None,
        contract_multiplier: Annotated[float, typer.Option(min=0.0)] = 1.0,
        lot_size: Annotated[float, typer.Option(min=0.0)] = 1.0,
        tick_size: Annotated[float | None, typer.Option(min=0.0)] = None,
        tick_value: Annotated[float | None, typer.Option(min=0.0)] = None,
        instrument_registry_path: Annotated[Path | None, typer.Option()] = None,
        calendar_path: Annotated[Path | None, typer.Option()] = None,
        session_filter: Annotated[str, typer.Option()] = "all",
        commission_bps: Annotated[float, typer.Option(min=0.0)] = 0.0,
        slippage_bps: Annotated[float, typer.Option(min=0.0)] = 0.0,
        slippage_ticks: Annotated[float, typer.Option(min=0.0)] = 0.0,
        bid_ask_spread_bps: Annotated[float, typer.Option(min=0.0)] = 0.0,
        bid_ask_spread_ticks: Annotated[float, typer.Option(min=0.0)] = 0.0,
        volume_participation_limit: Annotated[float | None, typer.Option(min=0.0)] = None,
        volume_to_contracts_multiplier: Annotated[float, typer.Option(min=0.0)] = 1.0,
        partial_fill_policy: Annotated[str, typer.Option()] = "allow",
        intrabar_ambiguity_policy: Annotated[str, typer.Option()] = "conservative",
        roll_cost_bps: Annotated[float, typer.Option(min=0.0)] = 0.0,
        roll_slippage_ticks: Annotated[float, typer.Option(min=0.0)] = 0.0,
        roll_pnl_policy: Annotated[str, typer.Option()] = "reject",
        output_root: Annotated[Path, typer.Option()] = Path("data/backtests"),
        run_id: Annotated[str, typer.Option(help="Deterministic run identifier.")] = ...,
        write_csv: Annotated[bool, typer.Option()] = False,
    ) -> None:
        try:
            event_config = build_event_config(locals())
            result = run_intraday_event_backtest(event_config)
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_dir={result.output_dir}")
        typer.echo(f"strategy_file={strategy_file}")
        typer.echo(f"event_count={result.metrics['event_count']}")
        typer.echo(f"strategy_call_count={result.metrics['strategy_call_count']}")
        typer.echo(f"trade_count={result.metrics['trade_count']}")
        typer.echo(f"total_return={result.metrics['total_return']:.12g}")
        typer.echo(f"max_drawdown={result.metrics['max_drawdown']:.12g}")


def build_event_config(values: dict[str, Any]) -> IntradayEventBacktestConfig:
    catalog_path = values["catalog_path"]
    dataset_id = values["dataset_id"]
    catalog_entry = (
        resolve_data_catalog_entry(catalog_path, dataset_id)
        if catalog_path is not None and dataset_id is not None
        else None
    )
    if values["dataset_path"] is None and catalog_entry is None:
        raise ValueError("Provide --dataset-path or both --catalog-path and --dataset-id")
    if values["dataset_path"] is not None and catalog_entry is not None:
        raise ValueError("Use either --dataset-path or --catalog-path/--dataset-id, not both")
    if values["strategy_file"] is None:
        raise ValueError("--strategy-file is required")
    dataset_path = (
        catalog_entry.dataset_path if catalog_entry is not None else values["dataset_path"]
    )
    quality_path = values["quality_approval_path"]
    registry_path = values["instrument_registry_path"]
    calendar_path = values["calendar_path"]
    if catalog_entry is not None:
        quality_path = quality_path or catalog_entry.quality_approval_path
        registry_path = registry_path or catalog_entry.instrument_registry_path
        calendar_path = calendar_path or catalog_entry.calendar_path
    return IntradayEventBacktestConfig(
        dataset=IntradayEventDatasetConfig(
            path=dataset_path,
            catalog_path=catalog_path,
            dataset_id=dataset_id,
            catalog_enforcement=values["catalog_enforcement"],
            quality_approval_path=quality_path,
            quality_approval_required=values["require_quality_approval"],
            min_usable_decision_day_pct=values["min_usable_decision_day_pct"],
            allow_legacy_bar_timestamps=values["allow_legacy_bar_timestamps"],
        ),
        strategy=IntradayEventStrategyConfig(
            strategy_file=values["strategy_file"],
            strategy_params=parse_params(values["param"] or []),
            strict_runtime=values.get("strict_runtime", False),
        ),
        instrument=IntradayEventInstrumentConfig(
            contract_multiplier=values["contract_multiplier"],
            lot_size=values["lot_size"],
            tick_size=values["tick_size"],
            tick_value=values["tick_value"],
            registry_path=registry_path,
        ),
        portfolio=IntradayEventPortfolioConfig(
            initial_capital=values["initial_capital"],
            sizing_mode=values["sizing_mode"],
            fixed_contracts=values["fixed_contracts"],
            target_notional=values["target_notional"],
        ),
        session=IntradayEventSessionConfig(
            calendar_path=calendar_path,
            session_filter=values["session_filter"],
        ),
        execution=IntradayEventExecutionConfig(
            timing=parse_intraday_event_timing(values["execution_timing"]),
            allow_noncausal_same_close=values["allow_noncausal_same_close"],
            end_of_data_position_policy=values["end_of_data_position_policy"],
            from_date=parse_date(values["from_date"]),
            schedule=parse_intraday_event_schedule(values["schedule"]),
            decision_times=[parse_time(value) for value in values["decision_time"] or []],
            decision_tolerance_minutes=values["decision_tolerance_minutes"],
            missing_decision_policy=parse_intraday_event_missing_decision_policy(
                values["missing_decision_policy"]
            ),
            commission_bps=values["commission_bps"],
            slippage_bps=values["slippage_bps"],
            slippage_ticks=values["slippage_ticks"],
            bid_ask_spread_bps=values["bid_ask_spread_bps"],
            bid_ask_spread_ticks=values["bid_ask_spread_ticks"],
            volume_participation_limit=values["volume_participation_limit"],
            volume_to_contracts_multiplier=values["volume_to_contracts_multiplier"],
            partial_fill_policy=values["partial_fill_policy"],
            intrabar_ambiguity_policy=values["intrabar_ambiguity_policy"],
            roll_cost_bps=values["roll_cost_bps"],
            roll_slippage_ticks=values["roll_slippage_ticks"],
            roll_pnl_policy=values["roll_pnl_policy"],
        ),
        output=IntradayEventOutputConfig(
            root=values["output_root"],
            run_id=values["run_id"],
            write_csv=values["write_csv"],
        ),
    )
