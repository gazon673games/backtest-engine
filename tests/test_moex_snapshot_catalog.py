from __future__ import annotations

from backtest_engine.adapters.moex.iss.catalog import list_moex_snapshot_futures_roots
from backtest_engine.adapters.moex.iss.snapshot_catalog import snapshot_instrument_rows


def test_local_snapshot_exposes_cny_rub_root_without_network() -> None:
    roots = list_moex_snapshot_futures_roots(query="CNY")

    assert [item["symbol"] for item in roots] == ["CR"]
    assert roots[0]["display_name"].startswith("CNY/RUB")
    assert roots[0]["contract_count"] >= 4


def test_snapshot_is_scoped_to_forts_futures() -> None:
    assert snapshot_instrument_rows(engine="stock", market="shares", board="TQBR") == []
