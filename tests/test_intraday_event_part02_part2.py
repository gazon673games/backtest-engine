# ruff: noqa: F405
from test_intraday_event_support import *  # noqa: F401,F403


def test_tick_slippage_and_roll_cost_are_reported_separately(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 2, 13, 40),
        ],
        closes=[100.0, 110.0, 120.0],
        active_contracts=["CRH4", "CRM4", "CRM4"],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        sizing_mode="notional",
        initial_capital=10_000.0,
        target_notional=1_000.0,
        tick_value=2.0,
        slippage_ticks=1.0,
        roll_slippage_ticks=2.0,
        roll_pnl_policy="exclude_gap",
    )
    row = events.row(0, named=True)

    assert row["slippage_cost_cash"] == pytest.approx(60.0)
    assert row["roll_cost_cash"] == pytest.approx(40.0)
    assert row["roll_turnover"] == pytest.approx(2.0)
    assert row["roll_contract_turnover"] == pytest.approx(20.0)
    assert row["turnover"] == pytest.approx(3.0)


def test_bid_ask_spread_is_reported_as_turnover_cost(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[datetime(2024, 1, 2, 13, 30), datetime(2024, 1, 2, 13, 35)],
        closes=[100.0, 110.0],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        bid_ask_spread_bps=20.0,
        bid_ask_spread_ticks=2.0,
        tick_size=0.5,
        sizing_mode="fixed_contracts",
        fixed_contracts=2.0,
        initial_capital=1000.0,
    )
    row = events.row(0, named=True)

    assert row["spread_cost_cash"] == pytest.approx(1.2)
    assert row["cost_cash"] == pytest.approx(1.2)
    assert row["gross_pnl"] == pytest.approx(20.0)
    assert row["net_pnl"] == pytest.approx(18.8)
