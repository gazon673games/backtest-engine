# ruff: noqa
from test_intraday_event_support import *  # noqa: F401,F403

from backtest_engine.intraday.event.futures import execution_costs


def test_roll_gap_requires_an_explicit_pnl_policy(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[datetime(2024, 1, 2, 13, 30), datetime(2024, 1, 2, 13, 35)],
        closes=[100.0, 200.0],
        active_contracts=["CRH4", "CRM4"],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    with pytest.raises(DatasetValidationError, match="Cannot calculate PnL across a futures roll"):
        run_intraday_event_backtest(make_config(tmp_path, dataset_path, plugin_path))


def test_explicit_roll_gap_exclusion_is_auditable_and_contract_specific(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 2, 13, 40),
        ],
        closes=[100.0, 200.0, 210.0],
        active_contracts=["CRH4", "CRM4", "CRM4"],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")
    result = run_intraday_event_backtest(
        make_config(
            tmp_path,
            dataset_path,
            plugin_path,
            roll_pnl_policy="exclude_gap",
        )
    )
    events = pl.read_parquet(result.output_dir / "events.parquet")
    trades = pl.read_parquet(result.output_dir / "trades.parquet")

    assert events.row(0, named=True)["contract_roll"] is True
    assert events.row(0, named=True)["roll_pnl_excluded"] is True
    assert events.row(0, named=True)["roll_pnl_policy"] == "exclude_gap"
    assert events.row(0, named=True)["gross_pnl"] == 0.0
    assert trades.height == 2
    assert trades.get_column("entry_active_contract").to_list() == ["CRH4", "CRM4"]
    assert trades.row(0, named=True)["gross_pnl"] == 0.0


@pytest.mark.parametrize("missing_field", ["approval_status", "dataset_sha256"])
def test_incomplete_quality_approval_is_rejected(
    tmp_path: Path,
    missing_field: str,
) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")
    approval_path = write_quality_approval(tmp_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    del approval[missing_field]
    approval_path.write_text(json.dumps(approval), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match=f"{missing_field} is required"):
        run_intraday_event_backtest(
            make_config(
                tmp_path,
                dataset_path,
                plugin_path,
                quality_approval_path=approval_path,
                quality_approval_required=True,
            )
        )


def test_intraday_dataset_requires_quality_approval_by_default(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)

    assert IntradayEventDatasetConfig(path=dataset_path).quality_approval_required is True


def test_quality_approval_cannot_approve_replaced_dataset(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")
    approval_path = write_quality_approval(tmp_path)
    pl.read_parquet(dataset_path).with_columns(
        (pl.col("volume") + 1.0).alias("volume")
    ).write_parquet(dataset_path)

    with pytest.raises(DatasetValidationError, match="dataset_sha256 mismatch"):
        run_intraday_event_backtest(
            make_config(
                tmp_path,
                dataset_path,
                plugin_path,
                quality_approval_path=approval_path,
                quality_approval_required=True,
            )
        )


def test_roll_turnover_notional_uses_old_close_and_new_open_prices() -> None:
    costs = execution_costs(
        row={"active_contract": "OLD"},
        next_row={"active_contract": "NEW"},
        current_contracts=1.0,
        contract_delta=0.0,
        turnover=0.0,
        turnover_notional=0.0,
        fill_price=100.0,
        exit_price=200.0,
        instrument=IntradayEventInstrumentSpec(price_currency="RUB", pnl_currency="RUB"),
        portfolio=IntradayEventPortfolioConfig(
            base_currency="RUB",
            initial_capital=1000.0,
            sizing_mode="fixed_contracts",
        ),
        execution=IntradayEventExecutionConfig(
            commission_bps=100.0,
            roll_pnl_policy="exclude_gap",
        ),
        previous_equity_cash=1000.0,
    )

    assert costs["roll_turnover_notional"] == pytest.approx(300.0)
    assert costs["commission_cost_cash"] == pytest.approx(3.0)


def test_session_filter_preserves_pnl_while_position_is_carried(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 10, 5),
            datetime(2024, 1, 2, 19, 0),
            datetime(2024, 1, 2, 19, 5),
            datetime(2024, 1, 3, 10, 0),
            datetime(2024, 1, 3, 10, 5),
        ],
        closes=[100.0, 100.0, 200.0, 400.0, 800.0, 800.0],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        session_filter="main",
    )

    assert events.get_column("strategy_called").to_list() == [True, True, False, False, True]
    assert events.get_column("equity").to_list()[-1] == pytest.approx(8.0)
