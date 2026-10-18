from __future__ import annotations

from typing import NoReturn

import typer
from pydantic import ValidationError

from backtest_engine.core.exceptions import BacktestError

CLI_EXCEPTIONS = (BacktestError, ValidationError, ValueError, KeyError)


def raise_cli_error(exc: Exception) -> NoReturn:
    typer.secho(str(exc), fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1) from exc
