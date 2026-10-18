from __future__ import annotations

from pathlib import Path

from backtest_engine.adapters.conformance import assert_adapter_conforms
from backtest_engine.adapters.models import (
    AdapterCapabilities,
    AdapterInfo,
    AdapterItem,
    DatasetBuildRequest,
    DatasetBuildResult,
)
from backtest_engine.adapters.registry import (
    get_adapter,
    list_market_adapters,
    register_adapter_factory,
    unregister_adapter_factory,
)


class FixtureAdapter:
    id = "fixture"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            id=self.id,
            name="Fixture",
            description="offline fixture",
            capabilities=AdapterCapabilities(asset_classes=["crypto"], data_kinds=["bars"]),
        )

    def engines(self) -> list[AdapterItem]:
        return [AdapterItem(id="spot", name="Spot")]

    def markets(self, engine: str) -> list[AdapterItem]:
        return [AdapterItem(id=engine, name=engine)]

    def boards(self, engine: str, market: str) -> list[AdapterItem]:
        return [AdapterItem(id=market, name=market)]

    def instruments(self, **_kwargs: object) -> list[dict[str, object]]:
        return []

    def build_dataset(self, request: DatasetBuildRequest, observer=None) -> DatasetBuildResult:
        del request, observer
        raise NotImplementedError


def test_runtime_adapter_discovery_and_conformance() -> None:
    register_adapter_factory(FixtureAdapter)
    try:
        adapter = get_adapter("fixture")
        assert_adapter_conforms(adapter)
        assert adapter.info().capabilities.asset_classes == ["crypto"]
    finally:
        unregister_adapter_factory(FixtureAdapter)


def test_fixture_signature_remains_typed() -> None:
    request = DatasetBuildRequest(
        adapter_id="fixture",
        engine="spot",
        market="spot",
        board="spot",
        security="BTCUSDT",
        asset_class="crypto",
        from_date="2024-01-01",
        till_date="2024-01-02",
        output_root=Path("data"),
    )

    assert request.asset_class == "crypto"


def test_builtin_order_preserves_moex_as_workbench_default() -> None:
    assert [adapter.id for adapter in list_market_adapters()[:2]] == ["moex", "binance"]
