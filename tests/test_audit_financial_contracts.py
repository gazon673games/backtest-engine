from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.accounting.reconciliation import reconcile_ledger
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.intraday.event.config import IntradayEventExecutionConfig
from backtest_engine.intraday.event.execution_v3 import IntradayExecutionBookV3
from backtest_engine.reporting.nav import drawdown_values
from backtest_engine.reporting.tearsheet.analytics import drawdown_periods
from backtest_engine.workbench._report_builder import equity_frame


def rows():
    return [
        {
            "ts": datetime(2024, 1, 1, 10) + timedelta(minutes=index),
            "open": 100.0,
            "close": 100.0,
            "high": 100.0,
            "low": 100.0,
            "volume": 20.0,
        }
        for index in range(6)
    ]


def submit(book, quantity=10.0, **overrides):
    values = dict(
        rows=rows(),
        index=0,
        order_type="limit",
        limit_price=100.0,
        stop_price=None,
        time_in_force="gtc",
        expires_after_bars=None,
        trailing_offset=None,
        oco_group=None,
        order_id="buy",
        cancel_order_id=None,
        replace_order_id=None,
        take_profit_price=None,
        stop_loss_price=None,
        post_only=False,
        reduce_only=False,
        self_trade_prevention="none",
        previous_signal=0.0,
        previous_contracts=0.0,
        requested_signal=1.0,
        requested_contracts=quantity,
        base_fill_price=100.0,
    )
    values.update(overrides)
    return book.submit_and_process(**values)


def test_gtc_fills_remain_consistent_with_confirmed_position_and_brackets():
    book = IntradayExecutionBookV3(IntradayEventExecutionConfig(volume_participation_limit=0.1))
    result = submit(book, take_profit_price=110.0, stop_loss_price=90.0)
    assert result["target_contracts"] == 2.0
    assert result["target_signal"] == 0.2
    state = book.simulator.orders["buy"]
    assert (state.filled_quantity, state.remaining_quantity) == (2.0, 8.0)
    exits = [state for state in book.simulator.active_orders() if state.request.order_id != "buy"]
    assert [state.request.quantity for state in exits] == [2.0, 2.0]
    result = book.process_pending(rows=rows(), index=1, base_fill_price=100.0)
    assert result["target_contracts"] == 4.0
    assert book.simulator.orders["buy"].remaining_quantity == 6.0
    exits = [state for state in book.simulator.active_orders() if state.request.order_id != "buy"]
    assert [state.request.quantity for state in exits] == [4.0, 4.0]


def test_queue_uses_observed_volume_even_without_participation_cap():
    book = IntradayExecutionBookV3(IntradayEventExecutionConfig(queue_ahead_contracts=1000.0))
    assert submit(book)["filled"] is False
    assert book.simulator.orders["buy"].remaining_quantity == 10.0


@pytest.mark.parametrize(
    "fractional,policy,expected",
    [(False, "allow", 1.0), (True, "allow", 1.5), (True, "reject", 0.0)],
)
def test_volume_policy_is_applied_by_book(fractional, policy, expected):
    book = IntradayExecutionBookV3(
        IntradayEventExecutionConfig(volume_participation_limit=0.075, partial_fill_policy=policy),
        allow_fractional_contracts=fractional,
    )
    result = submit(book)
    assert book.simulator.orders["buy"].filled_quantity == expected
    assert (result["target_contracts"] if result["filled"] else 0.0) == expected


def test_l2_accounting_uses_vwap_of_current_fills(tmp_path: Path):
    path = tmp_path / "book.parquet"
    at = rows()[0]["ts"]
    pl.DataFrame(
        [
            {
                "ts": at,
                "sequence": 1,
                "update_type": "snapshot",
                "side": "bid",
                "price": 90.0,
                "size": 10.0,
            },
            {
                "ts": at,
                "sequence": 1,
                "update_type": "snapshot",
                "side": "ask",
                "price": 100.0,
                "size": 1.0,
            },
            {
                "ts": at,
                "sequence": 1,
                "update_type": "snapshot",
                "side": "ask",
                "price": 110.0,
                "size": 1.0,
            },
        ]
    ).write_parquet(path)
    book = IntradayExecutionBookV3(IntradayEventExecutionConfig(order_book_path=path))
    result = submit(book, quantity=2.0, order_type="market", limit_price=None)
    assert result["target_contracts"] == 2.0
    assert result["fill_price"] == 105.0
    assert result["target_contracts"] * result["fill_price"] == 210.0


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf")])
def test_reconciliation_rejects_invalid_monetary_values(bad):
    row = {
        "ts": datetime(2024, 1, 1),
        "symbol": "TEST",
        "currency": "USDT",
        "net_cash_flow": 0.0,
        "cash_balance": 1000.0,
        "equity_cash": 1000.0,
    }
    with pytest.raises(DatasetValidationError, match="null|finite"):
        reconcile_ledger(pl.DataFrame([{**row, "cash_balance": bad}]), pl.DataFrame([row]))


def test_reconciliation_cannot_skip_missing_money_columns():
    keys = {"ts": datetime(2024, 1, 1), "symbol": "TEST", "currency": "USDT"}
    with pytest.raises(DatasetValidationError, match="missing columns"):
        reconcile_ledger(pl.DataFrame([keys]), pl.DataFrame([{**keys, "cash_balance": 1000.0}]))


def test_initial_loss_is_included_in_all_drawdown_reports():
    events = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "next_ts": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "equity_cash": [900.0, 800.0],
            "equity": [0.9, 0.8],
            "net_pnl": [-100.0, -100.0],
        }
    )
    assert drawdown_periods(events)["max_drawdown"][0] == pytest.approx(-0.2)
    assert equity_frame(events)["drawdown"].to_list() == pytest.approx([-0.1, -0.2])
    assert drawdown_values([0.9, 0.8], 1.0) == pytest.approx([-0.1, -0.2])
