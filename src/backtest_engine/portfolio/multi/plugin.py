from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.portfolio.multi.context import MultiInstrumentStrategyContext
from backtest_engine.strategies.lifecycle import lifecycle_from_module, strategy_lifecycle
from backtest_engine.strategies.runtime import (
    StrategyMetadata,
    StrategyParamSpec,
    validate_strategy_params,
)


def load_multi_instrument_strategy_plugin(
    strategy_file: Path,
    *,
    require_metadata: bool = False,
) -> Callable[..., Any]:
    module = load_module(strategy_file)
    generate_targets = getattr(module, "generate_targets", None)
    if generate_targets is None:
        raise StrategyPluginError(
            f"Multi-instrument strategy plugin must define generate_targets: {strategy_file}"
        )
    if not callable(generate_targets):
        raise StrategyPluginError(f"generate_targets must be callable: {strategy_file}")
    metadata = metadata_from_module(module)
    if metadata is None and require_metadata:
        raise StrategyPluginError("Strict strategy runtime requires STRATEGY_METADATA")
    if metadata is not None:
        generate_targets.__strategy_param_specs__ = metadata.params
    generate_targets.__strategy_lifecycle__ = lifecycle_from_module(module)
    return generate_targets


def load_module(path: Path) -> ModuleType:
    if not path.exists():
        raise StrategyPluginError(f"Strategy plugin does not exist: {path}")
    module_name = f"backtest_multi_strategy_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise StrategyPluginError(f"Could not load strategy plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_portfolio_decisions(
    generate_targets: Callable[..., Any],
    ctx: MultiInstrumentStrategyContext,
    params: dict[str, str],
) -> dict[str, dict[str, Any]]:
    params = validated_params(generate_targets, params)
    lifecycle = strategy_lifecycle(generate_targets)
    if lifecycle is not None:
        ctx.state = lifecycle.state
    try:
        if strategy_accepts_params(generate_targets):
            raw = generate_targets(ctx, params)
        else:
            raw = generate_targets(ctx)
    except StrategyPluginError:
        raise
    except Exception as exc:
        raise StrategyPluginError(f"Multi-instrument strategy failed: {exc}") from exc
    return normalize_decisions(raw, ctx.asset_ids)


def strategy_accepts_params(generate_targets: Callable[..., Any]) -> bool:
    cached = getattr(generate_targets, "__multi_accepts_params__", None)
    if cached is not None:
        return bool(cached)
    accepts_params = len(inspect.signature(generate_targets).parameters) >= 2
    try:
        generate_targets.__multi_accepts_params__ = accepts_params
    except (AttributeError, TypeError):
        pass
    return accepts_params


def metadata_from_module(module: ModuleType) -> StrategyMetadata | None:
    raw_metadata = getattr(module, "STRATEGY_METADATA", None)
    raw_params = getattr(module, "STRATEGY_PARAMS", None)
    if raw_metadata is None and raw_params is None:
        return None
    payload = raw_metadata if isinstance(raw_metadata, dict) else {}
    params = payload.get("params", raw_params or ())
    return StrategyMetadata(
        name=str(payload.get("name", module.__name__)),
        version=str(payload.get("version", "1")),
        params=tuple(param_spec(item) for item in params),
    )


def param_spec(raw: Any) -> StrategyParamSpec:
    if isinstance(raw, StrategyParamSpec):
        return raw
    if not isinstance(raw, dict):
        raise StrategyPluginError("Strategy param specs must be dicts or StrategyParamSpec")
    return StrategyParamSpec(**raw)


def validated_params(
    generate_targets: Callable[..., Any],
    params: dict[str, str],
) -> dict[str, Any]:
    specs = getattr(generate_targets, "__strategy_param_specs__", None)
    if specs is None:
        return params
    try:
        return validate_strategy_params(params, tuple(specs))
    except Exception as exc:
        raise StrategyPluginError(f"Strategy params failed validation: {exc}") from exc


def normalize_decisions(raw: Any, asset_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise StrategyPluginError("generate_targets must return a dict")
    payload = raw.get("targets", raw)
    if not isinstance(payload, dict):
        raise StrategyPluginError("generate_targets targets must be a dict")
    return {
        normalized_asset_id: normalize_decision_value(normalized_asset_id, value)
        for normalized_asset_id, value in normalized_items(payload, asset_ids)
    }


def normalized_items(payload: dict[Any, Any], asset_ids: tuple[str, ...]) -> list[tuple[str, Any]]:
    items = []
    for asset_id, value in payload.items():
        normalized_asset_id = str(asset_id)
        if normalized_asset_id not in asset_ids:
            raise StrategyPluginError(f"generate_targets returned unknown asset_id: {asset_id}")
        items.append((normalized_asset_id, value))
    return items


def normalize_decision_value(asset_id: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        raw_target = value.get("target", value.get("signal", value.get("target_position")))
        reason = value.get("reason")
        features = value.get("features", {})
    else:
        raw_target = value
        reason = None
        features = {}
    if raw_target is None:
        raise StrategyPluginError(f"Missing target for asset_id={asset_id}")
    if not isinstance(features, dict):
        raise StrategyPluginError(f"features must be a dict for asset_id={asset_id}")
    result = {
        "target": normalize_target(raw_target, asset_id),
        "reason": None if reason is None else str(reason),
        "features": features,
    }
    if isinstance(value, dict) and value.get("contracts") is not None:
        if result["target"] not in {-1, 0, 1}:
            raise StrategyPluginError(
                f"Explicit contracts require target -1, 0 or 1 for {asset_id}"
            )
        quantity = value["contracts"]
        if isinstance(quantity, bool) or not math.isfinite(float(quantity)) or float(quantity) < 0:
            raise StrategyPluginError(f"contracts must be finite and nonnegative for {asset_id}")
        if (result["target"] == 0) != (float(quantity) == 0):
            raise StrategyPluginError(f"contracts and target disagree for {asset_id}")
        result["contracts"] = float(quantity)
    return result


def normalize_target(raw_target: Any, asset_id: str) -> float:
    if isinstance(raw_target, bool):
        raise StrategyPluginError(f"target must be numeric for asset_id={asset_id}")
    try:
        value = float(raw_target)
    except (TypeError, ValueError) as exc:
        raise StrategyPluginError(f"target must be numeric for asset_id={asset_id}") from exc
    if not math.isfinite(value):
        raise StrategyPluginError(f"target must be finite for asset_id={asset_id}")
    if value < -1.0 or value > 1.0:
        raise StrategyPluginError(f"target must be between -1.0 and 1.0 for asset_id={asset_id}")
    return value
