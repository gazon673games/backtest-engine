# ruff: noqa: F405
from test_intraday_event_support import *  # noqa: F401,F403


def test_long_pnl_uses_close_to_next_close_return(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[datetime(2024, 1, 2, 13, 30), datetime(2024, 1, 2, 13, 35)],
        closes=[100.0, 110.0],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(tmp_path, dataset_path, plugin_path)

    assert events.get_column("asset_return").to_list() == pytest.approx([0.1])
    assert events.get_column("gross_return").to_list() == pytest.approx([0.1])
    assert events.get_column("equity").to_list() == pytest.approx([1.1])


def test_short_pnl_uses_negative_exposure(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[datetime(2024, 1, 2, 13, 30), datetime(2024, 1, 2, 13, 35)],
        closes=[100.0, 90.0],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return -1.0\n")

    events = build_events(tmp_path, dataset_path, plugin_path)

    assert events.get_column("asset_return").to_list() == pytest.approx([-0.1])
    assert events.get_column("gross_return").to_list() == pytest.approx([0.1])
    assert events.get_column("equity").to_list() == pytest.approx([1.1])


def test_long_to_short_turnover_is_two(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 2, 13, 40),
        ],
        closes=[100.0, 101.0, 102.0],
    )
    plugin_path = write_intraday_plugin(
        tmp_path,
        """
def generate_signal(ctx):
    return 1.0 if float(ctx.current_bar()["close"]) < 101.0 else -1.0
""".lstrip(),
    )

    events = build_events(tmp_path, dataset_path, plugin_path)

    assert events.get_column("target_position").to_list() == [1.0, -1.0]
    assert events.get_column("turnover").to_list() == pytest.approx([1.0, 2.0])


def test_short_to_long_turnover_is_two(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 2, 13, 40),
        ],
        closes=[100.0, 101.0, 102.0],
    )
    plugin_path = write_intraday_plugin(
        tmp_path,
        """
def generate_signal(ctx):
    return -1.0 if float(ctx.current_bar()["close"]) < 101.0 else 1.0
""".lstrip(),
    )

    events = build_events(tmp_path, dataset_path, plugin_path)

    assert events.get_column("target_position").to_list() == [-1.0, 1.0]
    assert events.get_column("turnover").to_list() == pytest.approx([1.0, 2.0])


def test_costs_are_applied_to_turnover(tmp_path: Path) -> None:
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

    events = build_events(tmp_path, dataset_path, plugin_path, commission_bps=10.0)

    assert events.get_column("turnover").to_list() == pytest.approx([1.0, 1.0])
    assert events.get_column("cost").to_list() == pytest.approx([0.001, 0.001])
    assert events.get_column("net_return").to_list() == pytest.approx([0.099, -0.001])


def test_notional_sizing_uses_contract_multiplier_and_cash_accounting(tmp_path: Path) -> None:
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
        contract_multiplier=10.0,
        initial_capital=10_000.0,
        sizing_mode="notional",
        target_notional=5_000.0,
    )
    row = events.row(0, named=True)

    assert row["target_contracts"] == 5.0
    assert row["position_notional"] == pytest.approx(5_000.0)
    assert row["gross_pnl"] == pytest.approx(500.0)
    assert row["net_pnl"] == pytest.approx(500.0)
    assert row["equity_cash"] == pytest.approx(10_500.0)
    assert row["equity"] == pytest.approx(1.05)


def test_strategy_context_exposes_equity_and_point_value_for_dynamic_sizing(
    tmp_path: Path,
) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[datetime(2024, 1, 2, 13, 30), datetime(2024, 1, 2, 13, 35)],
        closes=[100.0, 110.0],
    )
    plugin_path = write_intraday_plugin(
        tmp_path,
        """
def generate_signal(ctx):
    contracts = round(ctx.equity_cash * 2.5 / (ctx.current_bar()["close"] * ctx.point_value))
    return {"target": 1, "contracts": contracts, "features": {
        "equity": ctx.equity_cash, "point_value": ctx.point_value}}
""".lstrip(),
    )

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        contract_multiplier=10.0,
        initial_margin=100.0,
        initial_capital=10_000.0,
        sizing_mode="strategy_contracts",
        allow_fractional_contracts=False,
    )
    row = events.row(0, named=True)

    assert row["target_contracts"] == 25.0
    assert row["position_notional"] == 25_000.0
    assert row["strategy_features"] == '{"equity":10000.0,"point_value":10.0}'


def test_strategy_dict_decision_populates_reason_and_features(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_intraday_plugin(
        tmp_path,
        """
def generate_signal(ctx):
    return {"target": 1.0, "reason": "unit_test_signal", "features": {"close": 100}}
""".lstrip(),
    )

    result = run_intraday_event_backtest(make_config(tmp_path, dataset_path, plugin_path))
    actions = pl.read_parquet(result.output_dir / "decision_actions.parquet")
    row = actions.row(0, named=True)

    assert row["reason"] == "unit_test_signal"
    assert row["strategy_features"] == '{"close":100}'


def test_instrument_registry_selects_contract_specific_specs(tmp_path: Path) -> None:
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
        sizing_mode="notional",
        initial_capital=10_000.0,
        target_notional=1_000.0,
        instrument_specs=[
            IntradayEventInstrumentSpec(
                symbol="CR*",
                contract_multiplier=10.0,
                lot_size=1.0,
                tick_size=0.01,
            )
        ],
    )
    row = events.row(0, named=True)

    assert row["instrument_symbol"] == "CR*"
    assert row["point_value"] == 10.0
    assert row["target_contracts"] == 1.0


def test_risk_policy_clip_limits_notional_exposure(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        sizing_mode="notional",
        initial_capital=10_000.0,
        target_notional=10_000.0,
        max_position_notional=1_000.0,
        risk_policy="clip",
    )

    assert events.row(0, named=True)["position_notional"] == pytest.approx(1_000.0)


def test_risk_policy_clip_limits_margin_usage(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        sizing_mode="notional",
        initial_capital=10_000.0,
        target_notional=10_000.0,
        initial_margin=1_000.0,
        max_margin_usage=0.2,
        risk_policy="clip",
    )
    row = events.row(0, named=True)

    assert row["target_contracts"] == pytest.approx(2.0)
    assert row["position_notional"] == pytest.approx(200.0)
