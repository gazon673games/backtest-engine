from datetime import datetime
from decimal import Decimal

import pytest

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.portfolio.multi.capital import SleeveCapital
from backtest_engine.portfolio.multi.plugin import normalize_decision_value
from backtest_engine.portfolio.multi.strategy_config import MultiInstrumentStrategyConfig
from backtest_engine.strategies.sdk_numbers import native_quantity


def test_profit_transfer_preserves_losses_and_does_not_donate_open_gains():
    book = SleeveCapital({"A": 300, "B": 300}, 700, "monthly_profit")
    book.redistribute(datetime(2025, 1, 1), {"A": 0, "B": 0}, 700)
    book.settle(
        [
            {"asset_id": "A", "next_ts": datetime(2025, 2, 1), "net_pnl_base": 100},
            {"asset_id": "B", "next_ts": datetime(2025, 2, 1), "net_pnl_base": -20},
        ],
        780,
    )
    book.redistribute(datetime(2025, 2, 1), {"A": 1, "B": 0}, 780)
    assert [b.equity for b in book.budgets.values()] == [400, 280]
    assert book.transfers == []
    book.redistribute(datetime(2025, 3, 1), {"A": 0, "B": 0}, 780)
    assert [b.equity for b in book.budgets.values()] == [350, 330]
    assert [b.basis for b in book.budgets.values()] == [350, 350]
    assert sum(t["amount"] for t in book.transfers) == 0
    book.redistribute(datetime(2025, 4, 1), {"A": 0, "B": 0}, 780)
    assert len(book.transfers) == 2


def test_rebalance_changes_budgets_without_creating_portfolio_pnl():
    book = SleeveCapital({"A": 400, "B": 200}, 700, "monthly_rebalance")
    book.redistribute(datetime(2025, 1, 1), {"A": 1, "B": 0}, 700)
    book.budgets["A"].equity += 90
    book.redistribute(datetime(2025, 2, 1), {"A": 1, "B": 0}, 790)
    assert book.budgets["A"].equity == 460
    assert book.budgets["B"].equity == 230
    assert book.reserve == 100


@pytest.mark.parametrize("monthly_pnl", [140, -140])
def test_total_rebalance_preserves_all_weights_including_reserve(monthly_pnl):
    book = SleeveCapital({"ETH": 250, "XLM": 200, "BTC": 150}, 700, "monthly_rebalance_total")
    positions = {"ETH": 1, "XLM": 0, "BTC": -1}
    book.redistribute(datetime(2025, 1, 1), positions, 700)
    book.settle(
        [{"asset_id": "ETH", "next_ts": datetime(2025, 2, 1), "net_pnl_base": monthly_pnl}],
        700 + monthly_pnl,
    )
    assert book.reserve == 100
    book.redistribute(datetime(2025, 2, 1), positions, 700 + monthly_pnl)
    for asset, weight in (("ETH", 5), ("XLM", 4), ("BTC", 3)):
        assert book.budgets[asset].equity == pytest.approx((700 + monthly_pnl) * weight / 14)
    assert book.reserve == pytest.approx((700 + monthly_pnl) * 2 / 14)
    assert sum(t["amount"] for t in book.transfers) == pytest.approx(0)
    transfers = list(book.transfers)
    book.redistribute(datetime(2025, 2, 2), positions, 700 + monthly_pnl)
    assert book.transfers == transfers


def test_capital_fails_on_future_settlement_and_reconciliation_break():
    book = SleeveCapital({"A": 600}, 700, "independent")
    book.settle([{"asset_id": "A", "next_ts": datetime(2025, 2, 1), "net_pnl_base": 10}], 710)
    with pytest.raises(StrategyPluginError, match="future settlement"):
        book.redistribute(datetime(2025, 1, 31), {"A": 0}, 710)
    with pytest.raises(StrategyPluginError, match="does not reconcile"):
        book.reconcile(711)


@pytest.mark.parametrize("quantity", [-1, float("nan"), float("inf"), True])
def test_multi_explicit_contracts_reject_invalid_quantity(quantity):
    with pytest.raises(StrategyPluginError):
        normalize_decision_value("A", {"target": 1, "contracts": quantity})


def test_multi_quantity_is_explicit_and_zero_target_must_close():
    assert normalize_decision_value("A", {"target": -1, "contracts": 0.125})["contracts"] == 0.125
    with pytest.raises(StrategyPluginError, match="disagree"):
        normalize_decision_value("A", {"target": 0, "contracts": 1})


def test_multi_sdk_configuration_requires_one_source():
    with pytest.raises(ValueError, match="Choose"):
        MultiInstrumentStrategyConfig()
    with pytest.raises(ValueError, match="requires sdk_sleeves"):
        MultiInstrumentStrategyConfig(strategy_file="old.py", capital_policy="monthly_profit")


def test_native_quantity_removes_float_noise_without_repairing_real_inventory():
    assert native_quantity(0.03 + 0.01, 0.001) == Decimal(".04")
    assert native_quantity(0.048999999999999996, 0.001) == Decimal(".049")
    assert native_quantity(0.0491, 0.001) == Decimal(".0491")
