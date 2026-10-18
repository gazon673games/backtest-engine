from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from backtest_engine.cli.common import CLI_EXCEPTIONS, raise_cli_error
from backtest_engine.paper.engine import run_paper_feed
from backtest_engine.paper.feed import ParquetPaperFeed
from backtest_engine.paper.models import PaperTradingConfig


def register(app: typer.Typer) -> None:
    @app.command(name="paper-replay")
    def paper_replay(
        config_path: Annotated[Path, typer.Option()],
        events_path: Annotated[Path, typer.Option()],
        output_path: Annotated[Path, typer.Option()],
        max_events: Annotated[int | None, typer.Option(min=1)] = None,
    ) -> None:
        try:
            config = PaperTradingConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
            result = run_paper_feed(
                config,
                ParquetPaperFeed(events_path),
                max_events=max_events,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_path={output_path}")
        typer.echo(f"events={result.event_count}")
        typer.echo(f"fills={result.fill_count}")
        typer.echo(f"equity={result.final_equity_cash}")
