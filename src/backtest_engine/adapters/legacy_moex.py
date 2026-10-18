"""Lazy compatibility boundary for the explicitly requested legacy MOEX import."""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    return getattr(import_module("backtest_engine.adapters.moex.adapter"), name)
