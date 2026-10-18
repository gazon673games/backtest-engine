"""Enforces strategy metadata, typed params, warmup, and lookahead-safe access."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl
from strategy_sdk import ParameterSpec, StrategyError
from strategy_sdk import validate_strategy_params as sdk_validate_params

from backtest_engine.core.exceptions import DatasetValidationError

ParamType = Literal["int", "float", "decimal", "str", "bool"]


@dataclass(frozen=True)
class StrategyParamSpec:
    name: str
    param_type: ParamType
    default: Any = None
    required: bool = False
    min_value: float | None = None
    max_value: float | None = None
    choices: tuple[Any, ...] = ()


@dataclass(frozen=True)
class StrategyMetadata:
    name: str
    version: str = "1"
    description: str | None = None
    warmup_bars: int = 0
    params: tuple[StrategyParamSpec, ...] = ()
    timeframe: str | None = None
    schedule: str = "every-bar"
    decision_times: tuple[str, ...] = ()
    decision_tolerance_minutes: int | None = None
    missing_decision_policy: str = "carry"
    history_bars: int = 5000
    data_columns: tuple[str, ...] = ()
    decision_time_param: str | None = None


@dataclass(frozen=True)
class DecisionExplanation:
    target: float
    reason: str
    features: dict[str, Any]
    blocked_by: str | None = None


class StrategyRuntimeContext:
    def __init__(
        self,
        frame: pl.DataFrame,
        *,
        current_index: int,
        warmup_bars: int = 0,
    ) -> None:
        if current_index < 0 or current_index >= frame.height:
            raise DatasetValidationError("current_index is outside the frame")
        self._frame = frame
        self.current_index = current_index
        self.warmup_bars = warmup_bars

    def visible_frame(self) -> pl.DataFrame:
        return self._frame.slice(0, self.current_index + 1)

    def require_warmup(self) -> None:
        if self.current_index + 1 < self.warmup_bars:
            raise DatasetValidationError(
                f"Strategy warmup not satisfied: {self.current_index + 1} < {self.warmup_bars}"
            )

    def value(self, column: str, *, offset: int = 0) -> Any:
        index = self.current_index + offset
        if index > self.current_index:
            raise DatasetValidationError("Strategy attempted to read future data")
        if index < 0:
            raise DatasetValidationError("Strategy attempted to read before available history")
        return self._frame.get_column(column)[index]

    def series(self, column: str) -> pl.Series:
        return self.visible_frame().get_column(column)


def validate_strategy_params(
    raw_params: dict[str, Any],
    specs: tuple[StrategyParamSpec, ...],
) -> dict[str, Any]:
    unknown = set(raw_params) - {spec.name for spec in specs}
    if unknown:
        raise DatasetValidationError(f"Unknown strategy params: {', '.join(sorted(unknown))}")
    try:
        declarations = tuple(
            ParameterSpec(
                spec.name,
                spec.param_type,
                default=spec.default,
                required=spec.required,
                min_value=spec.min_value,
                max_value=spec.max_value,
                choices=spec.choices,
            )
            for spec in specs
        )
        return sdk_validate_params(raw_params, declarations)
    except StrategyError as error:
        raise DatasetValidationError(f"Strategy params failed validation: {error}") from error


def strategy_fingerprint(
    *,
    strategy_path: Path | None = None,
    metadata: StrategyMetadata | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    digest = hashlib.sha256()
    if strategy_path is not None and strategy_path.exists():
        digest.update(strategy_path.read_bytes())
    if metadata is not None:
        digest.update(
            json.dumps(_metadata_payload(metadata), sort_keys=True, default=str).encode("utf-8")
        )
    if params is not None:
        digest.update(json.dumps(params, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _metadata_payload(metadata: StrategyMetadata) -> dict[str, Any]:
    payload = {
        "name": metadata.name,
        "version": metadata.version,
        "description": metadata.description,
        "warmup_bars": metadata.warmup_bars,
        "params": [spec.__dict__ for spec in metadata.params],
        "timeframe": metadata.timeframe,
        "schedule": metadata.schedule,
        "decision_times": list(metadata.decision_times),
        "decision_tolerance_minutes": metadata.decision_tolerance_minutes,
        "missing_decision_policy": metadata.missing_decision_policy,
        "history_bars": metadata.history_bars,
        "data_columns": list(metadata.data_columns),
    }
    if metadata.decision_time_param is not None:
        payload["decision_time_param"] = metadata.decision_time_param
    return payload
