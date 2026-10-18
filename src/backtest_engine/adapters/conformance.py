"""Reusable adapter contract checks for built-ins and third-party packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backtest_engine.adapters.protocol import MarketDataAdapter


@dataclass(frozen=True)
class AdapterConformanceIssue:
    field: str
    message: str


def adapter_conformance_issues(adapter: MarketDataAdapter) -> list[AdapterConformanceIssue]:
    issues = []
    required_methods = ("info", "engines", "markets", "boards", "instruments", "build_dataset")
    for name in required_methods:
        if not callable(getattr(adapter, name, None)):
            issues.append(AdapterConformanceIssue(name, "required callable is missing"))
    if issues:
        return issues
    info = adapter.info()
    if not info.id.strip() or info.id != adapter.id:
        issues.append(AdapterConformanceIssue("id", "adapter.id must equal non-empty info().id"))
    if not info.capabilities.asset_classes:
        issues.append(
            AdapterConformanceIssue("asset_classes", "at least one asset class is required")
        )
    if not info.capabilities.data_kinds:
        issues.append(AdapterConformanceIssue("data_kinds", "at least one data kind is required"))
    if not info.capabilities.timeframes:
        issues.append(AdapterConformanceIssue("timeframes", "at least one timeframe is required"))
    issues.extend(item_issues("engines", adapter.engines()))
    return issues


def assert_adapter_conforms(adapter: MarketDataAdapter) -> None:
    issues = adapter_conformance_issues(adapter)
    if issues:
        detail = "; ".join(f"{issue.field}: {issue.message}" for issue in issues)
        adapter_id = getattr(adapter, "id", "<unknown>")
        raise ValueError(f"Adapter {adapter_id} violates contract: {detail}")


def item_issues(field: str, items: list[Any]) -> list[AdapterConformanceIssue]:
    if not items:
        return [AdapterConformanceIssue(field, "catalog must not be empty")]
    ids = [str(item.id).strip() for item in items]
    if any(not item_id for item_id in ids):
        return [AdapterConformanceIssue(field, "catalog item id must not be empty")]
    if len(ids) != len(set(ids)):
        return [AdapterConformanceIssue(field, "catalog item ids must be unique")]
    return []
