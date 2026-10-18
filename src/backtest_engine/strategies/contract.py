from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.lifecycle import lifecycle_from_module
from backtest_engine.strategies.runtime import (
    StrategyMetadata,
    StrategyParamSpec,
    strategy_fingerprint,
    validate_strategy_params,
)


def attach_strategy_contract(
    generate: Callable[..., Any],
    module: ModuleType,
    strategy_file: Path,
    *,
    require_metadata: bool = False,
) -> Callable[..., Any]:
    metadata = _strategy_metadata_from_module(module)
    if metadata is None and require_metadata:
        raise StrategyPluginError("Strict strategy runtime requires STRATEGY_METADATA")
    if metadata is not None:
        generate.__strategy_metadata__ = metadata
        generate.__strategy_param_specs__ = metadata.params
        generate.__strategy_warmup_bars__ = metadata.warmup_bars
    generate.__strategy_fingerprint__ = strategy_fingerprint(
        strategy_path=strategy_file,
        metadata=metadata,
        params=None,
    )
    generate.__strategy_lifecycle__ = lifecycle_from_module(module)
    return generate


def validated_strategy_params(
    generate: Callable[..., Any],
    params: dict[str, str],
) -> dict[str, Any]:
    if getattr(generate, "__strategy_sdk__", False):
        return params  # The pinned SDK instance validates its own schema once.
    specs = getattr(generate, "__strategy_param_specs__", None)
    if specs is None:
        return params
    try:
        return validate_strategy_params(params, tuple(specs))
    except Exception as exc:
        raise StrategyPluginError(f"Strategy params failed validation: {exc}") from exc


def enforce_intraday_warmup(generate: Callable[..., Any], ctx: IntradayStrategyContext) -> None:
    warmup_bars = int(getattr(generate, "__strategy_warmup_bars__", 0) or 0)
    if warmup_bars <= 0:
        return
    try:
        ctx.runtime(warmup_bars=warmup_bars).require_warmup()
    except Exception as exc:
        raise StrategyPluginError(f"Strategy warmup failed: {exc}") from exc


def _strategy_metadata_from_module(module: ModuleType) -> StrategyMetadata | None:
    raw_metadata = getattr(module, "STRATEGY_METADATA", None)
    raw_params = getattr(module, "STRATEGY_PARAMS", None)
    if raw_metadata is None and raw_params is None:
        return None
    if isinstance(raw_metadata, StrategyMetadata):
        return raw_metadata
    payload = raw_metadata if isinstance(raw_metadata, dict) else {}
    params = payload.get("params", raw_params or ())
    return StrategyMetadata(
        name=str(payload.get("name", module.__name__)),
        version=str(payload.get("version", "1")),
        description=payload.get("description"),
        warmup_bars=int(payload.get("warmup_bars", 0)),
        params=tuple(_param_spec(item) for item in params),
        timeframe=_optional_text(payload.get("timeframe")),
        schedule=str(payload.get("schedule", "every-bar")),
        decision_times=tuple(str(value) for value in payload.get("decision_times", ())),
        decision_tolerance_minutes=_optional_int(payload.get("decision_tolerance_minutes")),
        missing_decision_policy=str(payload.get("missing_decision_policy", "carry")),
        history_bars=int(payload.get("history_bars", 5000)),
        data_columns=tuple(str(value) for value in payload.get("data_columns", ())),
        decision_time_param=_optional_text(payload.get("decision_time_param")),
    )


def _param_spec(raw: Any) -> StrategyParamSpec:
    if isinstance(raw, StrategyParamSpec):
        return raw
    if not isinstance(raw, dict):
        raise StrategyPluginError("Strategy param specs must be dicts or StrategyParamSpec")
    return StrategyParamSpec(**raw)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _strategy_call_mode(generate: Callable[..., Any]) -> str:
    cached = getattr(generate, "__strategy_call_mode__", None)
    if cached is not None:
        return str(cached)
    parameters = list(inspect.signature(generate).parameters.values())
    has_var_args = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    )
    has_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    keyword_names = {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    if has_var_args or len(positional) >= 2:
        mode = "positional"
    elif has_var_kwargs or "params" in keyword_names:
        mode = "keyword"
    else:
        mode = "context"
    try:
        generate.__strategy_call_mode__ = mode
    except (AttributeError, TypeError):
        pass
    return mode
