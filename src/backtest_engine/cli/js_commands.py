from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from backtest_engine.cli.common import CLI_EXCEPTIONS, raise_cli_error
from backtest_engine.cli.event_command import build_event_config
from backtest_engine.intraday.event.engine import run_intraday_event_backtest


def register(app: typer.Typer) -> None:
    @app.command(name="run-js-strategy")
    def run_js_strategy(
        dataset_path: Annotated[Path | None, typer.Option()] = None,
        catalog_path: Annotated[Path | None, typer.Option()] = None,
        dataset_id: Annotated[str | None, typer.Option()] = None,
        catalog_enforcement: Annotated[str, typer.Option()] = "off",
        script: Annotated[Path, typer.Option("--script")] = ...,
        param: Annotated[list[str] | None, typer.Option("--param")] = None,
        strict_runtime: Annotated[bool, typer.Option()] = True,
        quality_approval_path: Annotated[Path | None, typer.Option()] = None,
        require_quality_approval: Annotated[bool, typer.Option()] = True,
        allow_legacy_bar_timestamps: Annotated[bool, typer.Option()] = False,
        allow_noncausal_same_close: Annotated[bool, typer.Option()] = False,
        end_of_data_position_policy: Annotated[str, typer.Option()] = "mark",
        min_usable_decision_day_pct: Annotated[float, typer.Option()] = 0.95,
        from_date: Annotated[str, typer.Option()] = "0001-01-01",
        schedule: Annotated[str, typer.Option()] = "times",
        decision_time: Annotated[list[str] | None, typer.Option("--decision-time")] = None,
        decision_tolerance_minutes: Annotated[int | None, typer.Option(min=0)] = None,
        missing_decision_policy: Annotated[str, typer.Option()] = "close",
        execution_timing: Annotated[str, typer.Option()] = "signal_close_to_next_bar_close",
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
        roll_pnl_policy: Annotated[str, typer.Option()] = "reject",
        output_root: Annotated[Path, typer.Option()] = Path("data/backtests"),
        run_id: Annotated[str, typer.Option(help="Deterministic run identifier.")] = ...,
        write_csv: Annotated[bool, typer.Option()] = True,
    ) -> None:
        try:
            config = build_event_config(_event_values(locals()))
            result = run_intraday_event_backtest(config)
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_dir={result.output_dir}")
        typer.echo(f"script={script}")
        typer.echo(f"event_count={result.metrics['event_count']}")
        typer.echo(f"strategy_call_count={result.metrics['strategy_call_count']}")
        typer.echo(f"trade_count={result.metrics['trade_count']}")
        typer.echo(f"total_return={result.metrics['total_return']:.12g}")
        typer.echo(f"max_drawdown={result.metrics['max_drawdown']:.12g}")


def _event_values(raw: dict[str, Any]) -> dict[str, Any]:
    values = dict(raw)
    values["strategy_file"] = values.pop("script")
    values.update(
        {
            "volume_participation_limit": None,
            "volume_to_contracts_multiplier": 1.0,
            "partial_fill_policy": "allow",
            "intrabar_ambiguity_policy": "conservative",
            "roll_cost_bps": 0.0,
            "roll_slippage_ticks": 0.0,
        }
    )
    return values
