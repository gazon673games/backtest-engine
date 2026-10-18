"""Publishes MOEX capabilities through the shared market-adapter contract."""

from __future__ import annotations

from typing import Any

from backtest_engine.adapters.build_events import DatasetBuildObserver
from backtest_engine.adapters.models import (
    AdapterCapabilities,
    AdapterInfo,
    AdapterItem,
    DatasetBuildRequest,
    DatasetBuildResult,
)
from backtest_engine.adapters.moex.iss.candles import build_moex_dataset
from backtest_engine.adapters.moex.iss.catalog import (
    list_moex_boards,
    list_moex_instruments,
    list_moex_markets,
    list_moex_snapshot_futures_roots,
)
from backtest_engine.adapters.moex.sessions import session_window_payloads
from backtest_engine.adapters.request_defaults import resolve_build_request


class MoexIssAdapter:
    id = "moex"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            id=self.id,
            name="MOEX ISS",
            description="Moscow Exchange ISS market data adapter.",
            capabilities=AdapterCapabilities(
                asset_classes=["futures"],
                data_kinds=["bars"],
                instrument_modes=["nearest_futures", "security"],
                default_engine="futures",
                default_instrument_mode="nearest_futures",
                instrument_catalogs={"nearest_futures": "futures-roots"},
                timeframes=["1m", "5m", "10m", "15m", "60m", "1d"],
                cache_policies=["prefer", "refresh", "offline", "network"],
                roll_rules=["expiry", "volume"],
                supports_offline=True,
                default_delay_minutes=15,
                provider_supports_realtime=True,
                realtime_requires_subscription=True,
                adapter_realtime_enabled=False,
                freshness_ttl_minutes=5,
                session_windows=session_window_payloads(),
                execution_defaults={
                    "trading_days_per_year": 252,
                    "commission_bps": 0.5,
                    "slippage_ticks": 1.0,
                    "bid_ask_spread_ticks": 1.0,
                    "execution_timing": "signal_close_to_next_bar_close",
                },
            ),
        )

    def engines(self) -> list[AdapterItem]:
        return [AdapterItem(id="futures", name="Futures")]

    def markets(self, engine: str) -> list[AdapterItem]:
        if engine == "futures":
            return [AdapterItem(id="forts", name="Derivatives market")]
        return list_moex_markets(engine)

    def boards(self, engine: str, market: str) -> list[AdapterItem]:
        if (engine, market) == ("futures", "forts"):
            return [AdapterItem(id="RFUD", name="Main futures board")]
        return list_moex_boards(engine, market)

    def instruments(
        self,
        *,
        engine: str,
        market: str,
        board: str,
        query: str,
    ) -> list[dict[str, Any]]:
        return list_moex_instruments(engine=engine, market=market, board=board, query=query)

    def futures_roots(
        self,
        *,
        engine: str,
        market: str,
        board: str,
        query: str,
    ) -> list[dict[str, Any]]:
        return list_moex_snapshot_futures_roots(
            engine=engine,
            market=market,
            board=board,
            query=query,
        )

    def build_dataset(
        self,
        request: DatasetBuildRequest,
        observer: DatasetBuildObserver | None = None,
    ) -> DatasetBuildResult:
        return build_moex_dataset(resolve_build_request(request, self), observer=observer)
