from __future__ import annotations

import math
from typing import Any

from backtest_engine.core.exceptions import StrategyPluginError


class StrategyParams:
    """Typed parameter reader for trusted strategy plugins."""

    def __init__(self, raw_params: dict[str, Any] | None = None) -> None:
        self._raw_params = raw_params or {}
        self._used: set[str] = set()

    def int(
        self,
        name: str,
        default: int | None = None,
        *,
        min_value: int | None = None,
        max_value: int | None = None,
        required: bool = False,
    ) -> int:
        value = self._value(name, default, required)
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise StrategyPluginError(f"Strategy parameter {name} must be an integer") from exc
        if min_value is not None and parsed < min_value:
            raise StrategyPluginError(f"Strategy parameter {name} must be >= {min_value}")
        if max_value is not None and parsed > max_value:
            raise StrategyPluginError(f"Strategy parameter {name} must be <= {max_value}")
        return parsed

    def float(
        self,
        name: str,
        default: float | None = None,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
        required: bool = False,
    ) -> float:
        value = self._value(name, default, required)
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise StrategyPluginError(f"Strategy parameter {name} must be numeric") from exc
        if not math.isfinite(parsed):
            raise StrategyPluginError(f"Strategy parameter {name} must be finite")
        if min_value is not None and parsed < min_value:
            raise StrategyPluginError(f"Strategy parameter {name} must be >= {min_value}")
        if max_value is not None and parsed > max_value:
            raise StrategyPluginError(f"Strategy parameter {name} must be <= {max_value}")
        return parsed

    def bool(self, name: str, default: bool | None = None, *, required: bool = False) -> bool:
        value = self._value(name, default, required)
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise StrategyPluginError(f"Strategy parameter {name} must be boolean")

    def str(
        self,
        name: str,
        default: str | None = None,
        *,
        choices: set[str] | frozenset[str] | None = None,
        required: bool = False,
    ) -> str:
        value = self._value(name, default, required)
        parsed = str(value)
        if choices is not None and parsed not in choices:
            raise StrategyPluginError(
                f"Strategy parameter {name} must be one of: {', '.join(sorted(choices))}"
            )
        return parsed

    def unused(self) -> set[str]:
        return set(self._raw_params).difference(self._used)

    def reject_unused(self) -> None:
        unused = sorted(self.unused())
        if unused:
            raise StrategyPluginError(f"Unknown strategy parameter(s): {', '.join(unused)}")

    def _value(self, name: str, default: Any, required: bool) -> Any:
        self._used.add(name)
        if name in self._raw_params and self._raw_params[name] is not None:
            return self._raw_params[name]
        if required:
            raise StrategyPluginError(f"Missing required strategy parameter: {name}")
        return default
