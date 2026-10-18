"""Explicit per-run state and lifecycle callbacks for strategy plugins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from backtest_engine.core.exceptions import StrategyPluginError


@dataclass
class StrategyLifecycle:
    initialize: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None
    on_fill: Callable[[dict[str, Any], dict[str, Any]], None] | None = None
    on_end: Callable[[dict[str, Any], dict[str, Any]], None] | None = None
    state: dict[str, Any] = field(default_factory=dict)
    started: bool = False
    finished: bool = False

    def start(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.started:
            return self.state
        self.started = True
        if self.initialize is None:
            return self.state
        try:
            initialized = self.initialize(dict(params))
        except Exception as exc:
            raise StrategyPluginError(f"Strategy initialize failed: {exc}") from exc
        if initialized is not None:
            if not isinstance(initialized, dict):
                raise StrategyPluginError("Strategy initialize must return a dict or None")
            self.state = initialized
        return self.state

    def fill(self, event: dict[str, Any]) -> None:
        if self.on_fill is None:
            return
        try:
            self.on_fill(self.state, dict(event))
        except Exception as exc:
            raise StrategyPluginError(f"Strategy on_fill failed: {exc}") from exc

    def finish(self, summary: dict[str, Any]) -> None:
        if self.finished:
            return
        self.finished = True
        if self.on_end is None:
            return
        try:
            self.on_end(self.state, dict(summary))
        except Exception as exc:
            raise StrategyPluginError(f"Strategy on_end failed: {exc}") from exc


def lifecycle_from_module(module: Any) -> StrategyLifecycle:
    return StrategyLifecycle(
        initialize=optional_callable(module, "initialize"),
        on_fill=optional_callable(module, "on_fill"),
        on_end=optional_callable(module, "on_end"),
    )


def optional_callable(module: Any, name: str) -> Callable[..., Any] | None:
    value = getattr(module, name, None)
    if value is not None and not callable(value):
        raise StrategyPluginError(f"Strategy lifecycle hook {name} must be callable")
    return value


def strategy_lifecycle(strategy: Any) -> StrategyLifecycle | None:
    value = getattr(strategy, "__strategy_lifecycle__", None)
    return value if isinstance(value, StrategyLifecycle) else None


def notify_strategy_fill(strategy: Any, event: dict[str, Any]) -> None:
    notifier = getattr(strategy, "notify_fill", None)
    if callable(notifier):
        notifier(event)
        return
    lifecycle = strategy_lifecycle(strategy)
    if lifecycle is not None:
        lifecycle.fill(event)


def finish_strategy(strategy: Any, summary: dict[str, Any]) -> None:
    finisher = getattr(strategy, "finish", None)
    if callable(finisher):
        finisher(summary)
        return
    lifecycle = strategy_lifecycle(strategy)
    if lifecycle is not None:
        lifecycle.finish(summary)
