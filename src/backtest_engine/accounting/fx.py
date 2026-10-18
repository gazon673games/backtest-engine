from __future__ import annotations

from collections import deque

from backtest_engine.core.exceptions import DatasetValidationError


def currency_code(value: str | None, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{field} must identify an explicit currency")
    return value.strip().upper()


def resolve_fx_rate(
    source_currency: str,
    target_currency: str,
    rates: dict[tuple[str, str], float],
) -> float:
    """Resolve direct, inverse, or triangulated conversion without lookahead."""
    source = currency_code(source_currency, "source_currency")
    target = currency_code(target_currency, "target_currency")
    if source == target:
        return 1.0
    graph: dict[str, list[tuple[str, float]]] = {}
    for (left, right), rate in rates.items():
        if rate <= 0.0:
            continue
        left = left.strip().upper()
        right = right.strip().upper()
        graph.setdefault(left, []).append((right, rate))
        graph.setdefault(right, []).append((left, 1.0 / rate))
    queue = deque([(source, 1.0)])
    visited = {source}
    while queue:
        currency, accumulated = queue.popleft()
        for next_currency, rate in graph.get(currency, []):
            if next_currency in visited:
                continue
            converted = accumulated * rate
            if next_currency == target:
                return converted
            visited.add(next_currency)
            queue.append((next_currency, converted))
    raise DatasetValidationError(f"Missing FX conversion path for {source}/{target}")
