# ruff: noqa
from test_intraday_event_support import *  # noqa: F401,F403


def test_event_loop_does_not_expose_future_bars(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 2, 13, 40),
        ],
        closes=[100.0, 999.0, 110.0],
    )
    plugin_path = write_intraday_plugin(
        tmp_path,
        """
def generate_signal(ctx):
    return 1.0 if max(ctx.series("close").to_list()) < 500.0 else 0.0
""".lstrip(),
    )

    events = build_events(tmp_path, dataset_path, plugin_path)

    assert events.get_column("target_position").to_list() == [1.0, 0.0]
    assert events.get_column("strategy_called").to_list() == [True, True]


def test_every_bar_schedule_calls_plugin_on_each_eligible_interval(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 2, 13, 40),
            datetime(2024, 1, 2, 13, 45),
        ],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    result = run_intraday_event_backtest(make_config(tmp_path, dataset_path, plugin_path))
    events = pl.read_parquet(result.output_dir / "events.parquet")

    assert result.metrics["event_count"] == 3
    assert result.metrics["strategy_call_count"] == 3
    assert events.get_column("strategy_called").to_list() == [True, True, True]


def test_times_schedule_calls_plugin_only_at_configured_times(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 20),
            datetime(2024, 1, 2, 13, 25),
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
        ],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        schedule="times",
        decision_times=["13:30"],
        decision_tolerance_minutes=5,
    )

    called_rows = events.filter(pl.col("strategy_called"))
    assert called_rows.get_column("ts").to_list() == [datetime(2024, 1, 2, 13, 30)]
    assert events.get_column("target_position").to_list() == [0.0, 0.0, 1.0]


def test_times_schedule_respects_tolerance(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 20),
            datetime(2024, 1, 2, 13, 25),
            datetime(2024, 1, 2, 13, 35),
        ],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    strict_events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        schedule="times",
        decision_times=["13:30"],
        decision_tolerance_minutes=4,
    )
    tolerant_events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        schedule="times",
        decision_times=["13:30"],
        decision_tolerance_minutes=5,
        run_id="tolerant",
    )

    assert strict_events.get_column("strategy_called").to_list() == [False, False]
    assert tolerant_events.filter(pl.col("strategy_called")).get_column("ts").to_list() == [
        datetime(2024, 1, 2, 13, 25)
    ]


def test_missing_decision_policy_carry_preserves_existing_position(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=missing_decision_timestamps(),
        closes=[100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        schedule="times",
        decision_times=["13:30"],
        decision_tolerance_minutes=4,
        missing_decision_policy="carry",
    )

    assert events.get_column("target_position").to_list() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert events.get_column("missing_decision").to_list() == [
        False,
        False,
        False,
        False,
        False,
    ]


def test_missing_decision_policy_close_flattens_at_nearest_available_bar(
    tmp_path: Path,
) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=missing_decision_timestamps(),
        closes=[100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        schedule="times",
        decision_times=["13:30"],
        decision_tolerance_minutes=4,
        missing_decision_policy="close",
    )
    missing = events.filter(pl.col("missing_decision")).row(0, named=True)

    assert events.get_column("target_position").to_list() == [1.0, 1.0, 0.0, 0.0, 1.0]
    assert missing["ts"] == datetime(2024, 1, 3, 13, 20)
    assert missing["missing_decision_policy"] == "close"
    assert missing["missing_decision_time"] == "13:30"
    assert missing["signal_reason"] == "missing_decision_close"


def test_missing_decision_policy_skip_preserves_accounting_for_missing_date(
    tmp_path: Path,
) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=missing_decision_timestamps(),
        closes=[100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
    )
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        schedule="times",
        decision_times=["13:30"],
        decision_tolerance_minutes=4,
        missing_decision_policy="skip",
    )

    assert events.get_column("ts").to_list() == [
        datetime(2024, 1, 2, 13, 30),
        datetime(2024, 1, 2, 13, 35),
        datetime(2024, 1, 3, 13, 20),
        datetime(2024, 1, 3, 13, 35),
        datetime(2024, 1, 4, 13, 30),
    ]
    skipped_day = events.filter(pl.col("ts").dt.date() == date(2024, 1, 3))
    assert not bool(skipped_day.get_column("strategy_called").any())
    assert skipped_day.get_column("target_position").to_list() == [1.0, 1.0]


def test_event_target_accepts_short_flat_and_long(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 2, 13, 40),
            datetime(2024, 1, 2, 13, 45),
        ],
        closes=[100.0, 101.0, 102.0, 103.0],
    )
    plugin_path = write_intraday_plugin(
        tmp_path,
        """
def generate_signal(ctx):
    close = float(ctx.current_bar()["close"])
    if close < 101.0:
        return -1.0
    if close < 102.0:
        return 0.0
    return 1.0
""".lstrip(),
    )

    events = build_events(tmp_path, dataset_path, plugin_path)

    assert events.get_column("target_position").to_list() == [-1.0, 0.0, 1.0]
