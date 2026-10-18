# ruff: noqa
from test_intraday_event_support import *  # noqa: F401,F403


def test_volume_participation_limit_allows_partial_fill(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[datetime(2024, 1, 2, 13, 30), datetime(2024, 1, 2, 13, 35)],
        closes=[100.0, 110.0],
        volumes=[5.0, 1000.0],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        sizing_mode="fixed_contracts",
        fixed_contracts=5.0,
        initial_capital=1000.0,
        volume_participation_limit=0.5,
        allow_fractional_contracts=False,
    )
    row = events.row(0, named=True)

    assert row["requested_contracts"] == pytest.approx(5.0)
    assert row["target_contracts"] == pytest.approx(2.0)
    assert row["target_position"] == pytest.approx(0.4)
    assert row["requested_contract_delta"] == pytest.approx(5.0)
    assert row["filled_contract_delta"] == pytest.approx(2.0)
    assert row["fill_fraction"] == pytest.approx(0.4)
    assert row["volume_limit_contracts"] == pytest.approx(2.0)
    assert row["volume_limited"] is True
    assert row["order_status"] == "partial_expired"
    assert row["gross_pnl"] == pytest.approx(20.0)


def test_volume_participation_limit_can_reject_order(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[datetime(2024, 1, 2, 13, 30), datetime(2024, 1, 2, 13, 35)],
        closes=[100.0, 110.0],
        volumes=[4.0, 1000.0],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        sizing_mode="fixed_contracts",
        fixed_contracts=5.0,
        volume_participation_limit=0.5,
        partial_fill_policy="reject",
    )
    row = events.row(0, named=True)

    assert row["requested_contracts"] == pytest.approx(5.0)
    assert row["target_contracts"] == pytest.approx(0.0)
    assert row["filled_contract_delta"] == pytest.approx(0.0)
    assert row["fill_fraction"] == pytest.approx(0.0)
    assert row["order_status"] == "volume_rejected"
    assert row["order_filled"] is False
    assert row["signal_reason"] == "volume_rejected"


def test_stop_limit_intrabar_ambiguity_policy_is_explicit(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[datetime(2024, 1, 2, 13, 30), datetime(2024, 1, 2, 13, 35)],
        closes=[100.0, 100.0],
        highs=[100.0, 110.0],
        lows=[100.0, 90.0],
    )
    plugin_path = write_intraday_plugin(
        tmp_path,
        """
def generate_signal(ctx):
    return {
        "target": 1.0,
        "order": {"type": "stop_limit", "stop_price": 105.0, "limit_price": 95.0},
    }
""".lstrip(),
    )

    conservative = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        intrabar_ambiguity_policy="conservative",
        execution_timing="signal_close_to_next_bar_open_to_close",
    )
    optimistic = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        intrabar_ambiguity_policy="optimistic",
        execution_timing="signal_close_to_next_bar_open_to_close",
    )

    assert conservative.row(0, named=True)["order_status"] == "ambiguous_unfilled"
    assert conservative.row(0, named=True)["target_position"] == 0.0
    assert optimistic.row(0, named=True)["order_status"] == "filled"
    assert optimistic.row(0, named=True)["fill_price"] == pytest.approx(95.0)
    assert optimistic.row(0, named=True)["target_position"] == 1.0


def test_execution_timing_can_fill_on_next_bar_open(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[datetime(2024, 1, 2, 13, 30), datetime(2024, 1, 2, 13, 35)],
        closes=[100.0, 120.0],
        opens=[100.0, 110.0],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        execution_timing="signal_close_to_next_bar_open_to_close",
    )
    row = events.row(0, named=True)

    assert row["fill_ts"] == datetime(2024, 1, 2, 13, 35)
    assert row["fill_price"] == 110.0
    assert row["exit_price"] == 120.0
    assert row["asset_return"] == pytest.approx(120.0 / 110.0 - 1.0)


def test_execution_timing_can_fill_on_next_bar_close(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 2, 13, 40),
        ],
        closes=[100.0, 110.0, 121.0],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        execution_timing="signal_close_to_next_bar_close_to_following_close",
    )
    row = events.row(0, named=True)

    assert row["fill_ts"] == datetime(2024, 1, 2, 13, 35)
    assert row["fill_price"] == 110.0
    assert row["next_ts"] == datetime(2024, 1, 2, 13, 40)
    assert row["asset_return"] == pytest.approx(121.0 / 110.0 - 1.0)


def test_session_filter_limits_every_bar_strategy_calls_to_main_session(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 9, 55),
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 10, 5),
            datetime(2024, 1, 2, 19, 0),
        ],
        closes=[100.0, 101.0, 102.0, 103.0],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        session_filter="main",
    )

    assert events.get_column("ts").to_list() == [
        datetime(2024, 1, 2, 9, 55),
        datetime(2024, 1, 2, 10, 0),
        datetime(2024, 1, 2, 10, 5),
    ]
    assert events.get_column("strategy_called").to_list() == [False, True, True]
    assert events.get_column("session_label").to_list() == ["off_session", "main", "main"]


def test_calendar_holidays_and_short_day_policy_skip_events(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 3, 13, 30),
            datetime(2024, 1, 3, 13, 35),
            datetime(2024, 1, 4, 13, 30),
            datetime(2024, 1, 4, 13, 35),
        ],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        holiday_dates=[date(2024, 1, 3)],
        short_day_dates=[date(2024, 1, 4)],
        short_day_policy="skip",
    )

    assert events.height == 5
    assert events.get_column("strategy_called").to_list() == [True, True, False, False, False]


def test_completed_long_trade_attribution(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 2, 13, 40),
        ],
        closes=[100.0, 110.0, 120.0],
    )
    plugin_path = write_intraday_plugin(
        tmp_path,
        "def generate_signal(ctx):\n    return 1.0 if ctx.current_ts.minute == 30 else 0.0\n",
    )

    result = run_intraday_event_backtest(make_config(tmp_path, dataset_path, plugin_path))
    trades = pl.read_parquet(result.output_dir / "trades.parquet")

    assert trades.height == 1
    trade = trades.row(0, named=True)
    assert trade["direction"] == "long"
    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 110.0
    assert trade["holding_bars"] == 1
    assert trade["gross_return"] == pytest.approx(0.1)
    assert trade["closed_by_end_of_data"] is False


def test_completed_short_trade_attribution(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 2, 13, 40),
        ],
        closes=[100.0, 90.0, 80.0],
    )
    plugin_path = write_intraday_plugin(
        tmp_path,
        "def generate_signal(ctx):\n    return -1.0 if ctx.current_ts.minute == 30 else 0.0\n",
    )

    result = run_intraday_event_backtest(make_config(tmp_path, dataset_path, plugin_path))
    trades = pl.read_parquet(result.output_dir / "trades.parquet")

    assert trades.height == 1
    trade = trades.row(0, named=True)
    assert trade["direction"] == "short"
    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 90.0
    assert trade["holding_bars"] == 1
    assert trade["gross_return"] == pytest.approx(0.1)
    assert trade["closed_by_end_of_data"] is False


def test_event_outputs_are_written(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    result = run_intraday_event_backtest(
        make_config(tmp_path, dataset_path, plugin_path, write_csv=True)
    )

    for file_name in [
        "config.json",
        "dataset_validation.json",
        "quality_approval.json",
        "events.parquet",
        "events.csv",
        "decision_actions.parquet",
        "decision_actions.csv",
        "equity.parquet",
        "equity.csv",
        "trades.parquet",
        "trades.csv",
        "portfolio_ledger.parquet",
        "portfolio_ledger.csv",
        "metrics.json",
        "experiment_manifest.json",
    ]:
        assert (result.output_dir / file_name).exists()
    manifest = json.loads((result.output_dir / "experiment_manifest.json").read_text())
    assert manifest["dataset"]["sha256"]
    assert manifest["strategy"]["sha256"]
    assert manifest["outputs"]["events.parquet"]["sha256"]
    assert manifest["outputs"]["portfolio_ledger.parquet"]["sha256"]
