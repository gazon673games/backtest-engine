"""Resolve ingestion inputs through the selected adapter's declared catalog."""

from pathlib import Path

from backtest_engine.adapters.models import AdapterItem, DatasetBuildRequest
from backtest_engine.adapters.protocol import MarketDataAdapter


def catalog_choice(value: str | None, items: list[AdapterItem], field: str) -> str:
    choices = [item.id for item in items]
    if value is not None:
        if value not in choices:
            raise ValueError(f"{field}={value!r} is unavailable; choose from {choices}")
        return value
    if len(choices) != 1:
        raise ValueError(f"{field} is required; choose from {choices}")
    return choices[0]


def resolve_build_request(
    request: DatasetBuildRequest, adapter: MarketDataAdapter
) -> DatasetBuildRequest:
    info = adapter.info()
    if request.adapter_id != info.id:
        raise ValueError(f"adapter_id={request.adapter_id!r} does not match {info.id!r}")
    capabilities = info.capabilities
    if (
        request.funding_price_policy != "event_only"
        and request.funding_price_policy not in capabilities.funding_price_policies
    ):
        raise ValueError(f"Unsupported funding_price_policy for {info.id}")
    for field, choices in (
        ("timeframe", capabilities.timeframes),
        ("data_kind", capabilities.data_kinds),
        ("cache_policy", capabilities.cache_policies),
    ):
        if choices and getattr(request, field) not in choices:
            raise ValueError(
                f"{field}={getattr(request, field)!r} is unavailable for {info.id}: {choices}"
            )
    engine = catalog_choice(
        request.engine or capabilities.default_engine, adapter.engines(), "engine"
    )
    market = catalog_choice(request.market, adapter.markets(engine), "market")
    board = catalog_choice(request.board, adapter.boards(engine, market), "board")
    if request.instrument_mode not in capabilities.instrument_modes:
        raise ValueError(f"Unsupported instrument_mode={request.instrument_mode!r} for {info.id}")
    if request.roll_rule and request.roll_rule not in capabilities.roll_rules:
        raise ValueError(f"Unsupported roll_rule={request.roll_rule!r} for {info.id}")
    if Path(info.id).name != info.id or info.id in {".", ".."}:
        raise ValueError(f"Invalid adapter id for cache namespace: {info.id!r}")
    return request.model_copy(
        update={
            "engine": engine,
            "market": market,
            "board": board,
            "cache_dir": request.cache_dir or request.output_root.parent / "cache" / info.id,
            "source_delay_minutes": (
                request.source_delay_minutes
                if request.source_delay_minutes is not None
                else capabilities.default_delay_minutes or 0
            ),
            "roll_rule": request.roll_rule or next(iter(capabilities.roll_rules), None),
        }
    )
