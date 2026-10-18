"""Loads trusted Python or JavaScript plugins and attaches runtime contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

import polars as pl

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.strategies.api import StrategyContext
from backtest_engine.strategies.contract import (
    _strategy_call_mode,
    attach_strategy_contract,
    enforce_intraday_warmup,
    validated_strategy_params,
)
from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.lifecycle import strategy_lifecycle
from backtest_engine.strategies.plugins.normalization import (
    IntradayEventPluginDecision,
    normalize_intraday_event_decision,
    normalize_intraday_event_signal,
    normalize_intraday_signal,
    normalize_plugin_signal,
)
from backtest_engine.strategies.sdk_adapter import SDKReplayStrategy, sdk_plugin

__all__ = [
    "IntradayEventPluginDecision",
    "generate_intraday_event_plugin_decision_from_callable",
    "generate_intraday_event_plugin_signal",
    "generate_intraday_event_plugin_signal_from_callable",
    "generate_intraday_plugin_signal",
    "generate_intraday_plugin_signal_from_callable",
    "generate_plugin_signal",
    "load_intraday_strategy_plugin",
    "load_strategy_plugin",
    "normalize_intraday_event_decision",
    "normalize_intraday_event_signal",
    "normalize_intraday_signal",
    "normalize_plugin_signal",
    "strategy_file_sha256",
]


def strategy_file_sha256(strategy_file: Path) -> str:
    _validate_strategy_file(strategy_file, suffixes=(".py", ".js"))
    digest = hashlib.sha256()
    with strategy_file.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strategy_plugin(
    strategy_file: Path,
    *,
    require_metadata: bool = False,
    params: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    module = _load_module(strategy_file)
    sdk = sdk_plugin(module, strategy_file, daily=True, params=params)
    if sdk is not None:
        return sdk
    generate_signals = getattr(module, "generate_signals", None)
    if generate_signals is None:
        msg = f"Strategy plugin must define generate_signals: {strategy_file}"
        raise StrategyPluginError(msg)
    if not callable(generate_signals):
        msg = f"Strategy plugin generate_signals must be callable: {strategy_file}"
        raise StrategyPluginError(msg)
    return attach_strategy_contract(
        generate_signals,
        module,
        strategy_file,
        require_metadata=require_metadata,
    )


def load_intraday_strategy_plugin(
    strategy_file: Path,
    *,
    require_metadata: bool = False,
    params: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    if strategy_file.suffix.lower() == ".js":
        _validate_strategy_file(strategy_file, suffixes=(".js",))
        marker = re.fullmatch(
            r'export const SDK_STRATEGY = "([a-z][a-z0-9_]*)";',
            strategy_file.read_text(encoding="utf-8-sig").strip(),
        )
        if marker:
            return SDKReplayStrategy(marker[1], strategy_file, params=params)
        from backtest_engine.strategies.javascript.runtime import load_js_intraday_strategy_plugin

        return load_js_intraday_strategy_plugin(
            strategy_file,
            require_metadata=require_metadata,
        )
    module = _load_module(strategy_file)
    sdk = sdk_plugin(module, strategy_file, params=params)
    if sdk is not None:
        return sdk
    generate_signal = getattr(module, "generate_signal", None)
    if generate_signal is None:
        msg = f"Intraday strategy plugin must define generate_signal: {strategy_file}"
        raise StrategyPluginError(msg)
    if not callable(generate_signal):
        msg = f"Intraday strategy plugin generate_signal must be callable: {strategy_file}"
        raise StrategyPluginError(msg)
    return attach_strategy_contract(
        generate_signal,
        module,
        strategy_file,
        require_metadata=require_metadata,
    )


def generate_plugin_signal(
    df: pl.DataFrame,
    strategy_file: Path,
    params: dict[str, str] | None = None,
) -> list[float]:
    generate_signals = load_strategy_plugin(strategy_file, params=params)
    ctx = StrategyContext(df)
    raw_signal = _call_generate_signals(generate_signals, ctx, params or {})
    return normalize_plugin_signal(raw_signal, expected_length=df.height)


def generate_intraday_plugin_signal(
    ctx: IntradayStrategyContext,
    strategy_file: Path,
    params: dict[str, str] | None = None,
) -> float:
    generate_signal = load_intraday_strategy_plugin(strategy_file, params=params)
    return generate_intraday_plugin_signal_from_callable(generate_signal, ctx, params)


def generate_intraday_plugin_signal_from_callable(
    generate_signal: Callable[..., Any],
    ctx: IntradayStrategyContext,
    params: dict[str, str] | None = None,
) -> float:
    raw_signal = _call_generate_signal(generate_signal, ctx, params or {})
    if getattr(generate_signal, "__strategy_sdk__", False):
        raw_signal = raw_signal["target"]
    return normalize_intraday_signal(raw_signal)


def generate_intraday_event_plugin_signal(
    ctx: IntradayStrategyContext,
    strategy_file: Path,
    params: dict[str, str] | None = None,
) -> float:
    generate_signal = load_intraday_strategy_plugin(strategy_file, params=params)
    return generate_intraday_event_plugin_signal_from_callable(generate_signal, ctx, params)


def generate_intraday_event_plugin_signal_from_callable(
    generate_signal: Callable[..., Any],
    ctx: IntradayStrategyContext,
    params: dict[str, str] | None = None,
) -> float:
    return generate_intraday_event_plugin_decision_from_callable(
        generate_signal,
        ctx,
        params,
    ).target


def generate_intraday_event_plugin_decision_from_callable(
    generate_signal: Callable[..., Any],
    ctx: IntradayStrategyContext,
    params: dict[str, str] | None = None,
) -> IntradayEventPluginDecision:
    raw_signal = _call_generate_signal(generate_signal, ctx, params or {})
    return normalize_intraday_event_decision(raw_signal)


def _load_module(strategy_file: Path) -> ModuleType:
    _validate_strategy_file(strategy_file)
    source = strategy_file.read_bytes()
    module_name = f"_backtest_engine_strategy_{hashlib.sha256(source).hexdigest()}"
    import_name = f"{module_name}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(import_name, strategy_file)
    if spec is None or spec.loader is None:
        msg = f"Could not load strategy plugin: {strategy_file}"
        raise StrategyPluginError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[import_name] = module
    try:
        exec(compile(source, str(strategy_file), "exec"), module.__dict__)
        module.__strategy_source__ = source
    except Exception as exc:
        msg = f"Strategy plugin failed during import: {strategy_file}: {exc}"
        raise StrategyPluginError(msg) from exc
    finally:
        sys.modules.pop(import_name, None)
    module.__name__ = module_name
    return module


def _validate_strategy_file(
    strategy_file: Path,
    *,
    suffixes: tuple[str, ...] = (".py",),
) -> None:
    raw_path = str(strategy_file)
    suffix_list = ", ".join(suffixes)
    if "://" in raw_path:
        msg = f"Strategy plugin must be a local {suffix_list} file: {strategy_file}"
        raise StrategyPluginError(msg)
    if strategy_file.suffix.lower() not in suffixes:
        msg = f"Strategy plugin must be a {suffix_list} file: {strategy_file}"
        raise StrategyPluginError(msg)
    if not strategy_file.exists():
        msg = f"Strategy plugin file does not exist: {strategy_file}"
        raise StrategyPluginError(msg)
    if not strategy_file.is_file():
        msg = f"Strategy plugin path is not a file: {strategy_file}"
        raise StrategyPluginError(msg)


def _call_generate_signals(
    generate_signals: Callable[..., Any],
    ctx: StrategyContext,
    params: dict[str, str],
) -> Any:
    params = validated_strategy_params(generate_signals, params)
    call_mode = _strategy_call_mode(generate_signals)

    try:
        if call_mode == "positional":
            return generate_signals(ctx, params)
        if call_mode == "keyword":
            return generate_signals(ctx, params=params)
        return generate_signals(ctx)
    except StrategyPluginError:
        raise
    except Exception as exc:
        msg = f"Strategy plugin generate_signals failed: {exc}"
        raise StrategyPluginError(msg) from exc


def _call_generate_signal(
    generate_signal: Callable[..., Any],
    ctx: IntradayStrategyContext,
    params: dict[str, str],
) -> Any:
    params = validated_strategy_params(generate_signal, params)
    lifecycle = strategy_lifecycle(generate_signal)
    if lifecycle is not None:
        ctx.state = lifecycle.start(params)
    enforce_intraday_warmup(generate_signal, ctx)
    call_mode = _strategy_call_mode(generate_signal)

    try:
        if call_mode == "positional":
            return generate_signal(ctx, params)
        if call_mode == "keyword":
            return generate_signal(ctx, params=params)
        return generate_signal(ctx)
    except StrategyPluginError:
        raise
    except Exception as exc:
        msg = f"Intraday strategy plugin generate_signal failed: {exc}"
        raise StrategyPluginError(msg) from exc
