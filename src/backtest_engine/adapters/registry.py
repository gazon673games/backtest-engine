"""Discovers built-in, package entry-point, and explicitly registered adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from importlib import import_module, metadata
from importlib.util import find_spec

from backtest_engine.adapters.models import AdapterInfo
from backtest_engine.adapters.protocol import MarketDataAdapter

ADAPTER_ENTRY_POINT_GROUP = "backtest_engine.market_adapters"
AdapterFactory = Callable[[], MarketDataAdapter]
_registered_factories: list[AdapterFactory] = []
BUILTIN_ADAPTERS = {
    "moex": "backtest_engine.adapters.moex.adapter:MoexIssAdapter",
    "binance": "backtest_engine.adapters.binance.adapter:BinanceAdapter",
}
BUILTIN_REQUIREMENTS = {"moex": ("continuous_futures_builder", "moex_iss_adapter")}


def register_adapter_factory(factory: AdapterFactory) -> None:
    if factory not in _registered_factories:
        _registered_factories.append(factory)


def unregister_adapter_factory(factory: AdapterFactory) -> None:
    if factory in _registered_factories:
        _registered_factories.remove(factory)


def list_market_adapters() -> list[MarketDataAdapter]:
    adapters = [factory() for factory in builtin_factories()]
    adapters.extend(factory() for factory in _registered_factories)
    adapters.extend(load_entry_point_adapters())
    by_id: dict[str, MarketDataAdapter] = {}
    for adapter in adapters:
        adapter_id = adapter.info().id
        if adapter_id in by_id:
            raise ValueError(f"Duplicate market adapter id: {adapter_id}")
        by_id[adapter_id] = adapter
    # Порядок built-ins является частью UX: первый адаптер выбирается в Workbench.
    return list(by_id.values())


def builtin_factories() -> Iterable[AdapterFactory]:
    for adapter_id, reference in BUILTIN_ADAPTERS.items():
        if all(find_spec(name) is not None for name in BUILTIN_REQUIREMENTS.get(adapter_id, ())):
            yield partial(load_builtin, reference)


def load_builtin(reference: str) -> MarketDataAdapter:
    module, name = reference.split(":")
    return getattr(import_module(module), name)()


def load_entry_point_adapters() -> list[MarketDataAdapter]:
    result = []
    selected = metadata.entry_points().select(group=ADAPTER_ENTRY_POINT_GROUP)
    for entry_point in sorted(selected, key=lambda item: item.name):
        loaded = entry_point.load()
        adapter = loaded() if isinstance(loaded, type) or callable(loaded) else loaded
        result.append(adapter)
    return result


def list_adapters() -> list[AdapterInfo]:
    return [adapter.info() for adapter in list_market_adapters()]


def get_adapter(adapter_id: str) -> MarketDataAdapter:
    adapters = [factory() for factory in _registered_factories]
    adapters.extend(load_entry_point_adapters())
    if adapter_id in BUILTIN_ADAPTERS:
        if any(find_spec(name) is None for name in BUILTIN_REQUIREMENTS.get(adapter_id, ())):
            raise ValueError(
                f"Adapter {adapter_id} requires backtest-engine[{adapter_id}] installation"
            )
        adapters.append(load_builtin(BUILTIN_ADAPTERS[adapter_id]))
    matches = [adapter for adapter in adapters if adapter.info().id == adapter_id]
    if len(matches) > 1:
        raise ValueError(f"Duplicate market adapter id: {adapter_id}")
    if matches:
        return matches[0]
    raise ValueError(f"Unknown adapter: {adapter_id}")


def require_adapter(adapter_id: str) -> AdapterInfo:
    return get_adapter(adapter_id).info()
