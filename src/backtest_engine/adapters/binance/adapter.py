"""Binance spot and USD-M perpetual adapter implementation."""

from __future__ import annotations

from typing import Any

from backtest_engine.adapters.binance.download import BINANCE_TIMEFRAMES, canonical_timeframe
from backtest_engine.adapters.binance.history import binance_history_range
from backtest_engine.adapters.binance.pipeline import build_binance_dataset
from backtest_engine.adapters.binance.transport import (
    BinanceHttpTransport,
    BinanceTransport,
    endpoint,
)
from backtest_engine.adapters.build_events import DatasetBuildObserver
from backtest_engine.adapters.models import (
    AdapterCapabilities,
    AdapterInfo,
    AdapterItem,
    DatasetBuildRequest,
    DatasetBuildResult,
    InstrumentHistoryRange,
    InstrumentItem,
)
from backtest_engine.adapters.request_defaults import resolve_build_request


class BinanceAdapter:
    id = "binance"

    def __init__(self, transport: BinanceTransport | None = None) -> None:
        self.transport = transport or BinanceHttpTransport()

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            id=self.id,
            name="Binance",
            description="Public Binance spot and USD-M perpetual market data.",
            capabilities=AdapterCapabilities(
                asset_classes=["crypto"],
                data_kinds=["bars", "trades", "quotes", "book"],
                instrument_modes=["security"],
                default_engine="perpetual",
                timeframes=[canonical_timeframe(value) for value in BINANCE_TIMEFRAMES],
                cache_policies=["prefer", "network", "only"],
                archive_fallback_policies=[
                    "missing_only",
                    "invalid_timestamps",
                    "invalid_timestamps_and_missing_marks",
                ],
                default_archive_fallback_policy="invalid_timestamps_and_missing_marks",
                supports_offline=True,
                default_delay_minutes=0,
                provider_supports_realtime=True,
                adapter_realtime_enabled=False,
                freshness_ttl_minutes=1,
                supports_rule_archive=True,
                supports_history_range=True,
                supports_dataset_prefix_reuse=True,
                funding_price_policies=["event_only", "minute_mark_open_estimate"],
                default_rules_enforcement="audit",
                default_quality_approval_required=False,
                session_windows=[{"start": "00:00", "end": "23:59"}],
                execution_defaults={
                    "execution_timing": "signal_close_to_next_bar_close",
                    "liquidity_role": "taker",
                    "trading_days_per_year": 365,
                    "allow_fractional_contracts": "true",
                },
            ),
        )

    def engines(self) -> list[AdapterItem]:
        return [
            AdapterItem(id="spot", name="Spot"),
            AdapterItem(id="perpetual", name="USD-M Futures (perpetual)"),
        ]

    def markets(self, engine: str) -> list[AdapterItem]:
        return [AdapterItem(id="crypto", name="Crypto")] if valid_engine(engine) else []

    def boards(self, engine: str, market: str) -> list[AdapterItem]:
        if not valid_engine(engine) or market != "crypto":
            return []
        board = "SPOT" if engine == "spot" else "USD_M_PERPETUAL"
        return [AdapterItem(id=board, name=board.replace("_", " ").title())]

    def instruments(
        self,
        *,
        engine: str,
        market: str,
        board: str,
        query: str,
    ) -> list[dict[str, Any]]:
        del market, board
        if not valid_engine(engine):
            return []
        payload = self.transport.get(engine, endpoint(engine, "exchangeInfo"))
        needle = query.strip().upper()
        result = []
        for item in payload.get("symbols", []):
            if engine == "perpetual" and item.get("contractType", "PERPETUAL") != "PERPETUAL":
                continue
            haystack = " ".join(
                str(item.get(key, "")) for key in ("symbol", "baseAsset", "quoteAsset")
            ).upper()
            if needle and needle not in haystack:
                continue
            result.append(
                InstrumentItem(
                    symbol=str(item["symbol"]),
                    short_name=f"{item.get('baseAsset')}/{item.get('quoteAsset')}",
                    name=str(item["symbol"]),
                    board="SPOT" if engine == "spot" else "USD_M_PERPETUAL",
                    market="crypto",
                    engine=engine,
                    metadata={
                        "status": item.get("status"),
                        "base_asset": item.get("baseAsset"),
                        "quote_asset": item.get("quoteAsset"),
                    },
                ).model_dump()
            )
        return result

    def history_range(self, *, engine: str, symbol: str, timeframe: str) -> InstrumentHistoryRange:
        return binance_history_range(
            self.transport, engine=engine, symbol=symbol, timeframe=timeframe
        )

    def build_dataset(
        self,
        request: DatasetBuildRequest,
        observer: DatasetBuildObserver | None = None,
    ) -> DatasetBuildResult:
        return build_binance_dataset(resolve_build_request(request, self), self.transport, observer)


def valid_engine(engine: str) -> bool:
    return engine in {"spot", "perpetual"}
