"""Lazy public facade for the Typer command-line application."""

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    """Load the command composition root only when the CLI itself is requested."""
    if name != "app":
        raise AttributeError(name)
    from backtest_engine.cli.composition import app

    return app
