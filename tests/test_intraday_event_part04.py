# ruff: noqa
from test_intraday_event_support import *  # noqa: F401,F403
from backtest_engine.data.bar_availability import ensure_bar_availability


def test_decision_uses_last_closed_bar_not_bar_opening_at_decision(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 20),
            datetime(2024, 1, 2, 13, 25),
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
        ],
        closes=[10.0, 20.0, 999.0, 30.0],
    )
    ensure_bar_availability(pl.read_parquet(dataset_path), "5m").write_parquet(dataset_path)
    plugin_path = write_intraday_plugin(
        tmp_path,
        "def generate_signal(ctx):\n    return 1.0 if ctx.current_bar()['close'] == 20 else 0.0\n",
    )

    events = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        schedule="times",
        decision_times=["13:30"],
        decision_tolerance_minutes=0,
    )
    decision = events.filter(pl.col("strategy_called")).row(0, named=True)

    assert decision["ts"] == datetime(2024, 1, 2, 13, 30)
    assert decision["bar_open_ts"] == datetime(2024, 1, 2, 13, 25)
    assert decision["fill_ts"] == datetime(2024, 1, 2, 13, 30)
    assert decision["target_position"] == 1.0


def test_two_decision_times_cannot_silently_share_one_bar(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 25),
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
        ],
    )
    ensure_bar_availability(pl.read_parquet(dataset_path), "5m").write_parquet(dataset_path)
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    with pytest.raises(DatasetValidationError, match="map to the same bar"):
        build_events(
            tmp_path,
            dataset_path,
            plugin_path,
            schedule="times",
            decision_times=["13:30", "13:31"],
            decision_tolerance_minutes=1,
        )


def test_next_bar_close_has_one_full_bar_of_execution_latency(tmp_path: Path) -> None:
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

    row = build_events(
        tmp_path,
        dataset_path,
        plugin_path,
        execution_timing="signal_close_to_next_bar_close",
    ).row(0, named=True)

    assert row["fill_price"] == 110.0
    assert row["exit_price"] == 121.0
    assert row["asset_return"] == pytest.approx(0.1)


def test_quality_approval_rejects_run_before_recommended_start(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")
    approval_path = write_quality_approval(
        tmp_path,
        recommended_start_date="2024-02-01",
    )

    with pytest.raises(DatasetValidationError, match="Quality approval failed"):
        run_intraday_event_backtest(
            make_config(
                tmp_path,
                dataset_path,
                plugin_path,
                quality_approval_path=approval_path,
                quality_approval_required=True,
            )
        )


def test_quality_approval_is_written_when_approved(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")
    approval_path = write_quality_approval(tmp_path)

    result = run_intraday_event_backtest(
        make_config(
            tmp_path,
            dataset_path,
            plugin_path,
            quality_approval_path=approval_path,
            quality_approval_required=True,
        )
    )
    approval = json.loads((result.output_dir / "quality_approval.json").read_text())

    assert result.quality_approval["status"] == "approved"
    assert approval["status"] == "approved"


def test_catalog_audit_does_not_restore_disabled_quality_gate(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")
    approval_path = write_quality_approval(tmp_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["approval_status"] = "rejected"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "dataset_id": "fixture",
                        "dataset_path": str(dataset_path),
                        "quality_status": "rejected",
                        "quality_approval_path": str(approval_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = make_config(tmp_path, dataset_path, plugin_path)
    dataset = config.dataset.model_copy(
        update={
            "catalog_path": catalog_path,
            "dataset_id": "fixture",
            "catalog_enforcement": "audit",
        }
    )

    result = run_intraday_event_backtest(config.model_copy(update={"dataset": dataset}))

    assert result.quality_approval["status"] == "not_required"


def test_event_metrics_include_exposure_and_trade_counts(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 2, 13, 40),
            datetime(2024, 1, 2, 13, 45),
        ],
        closes=[100.0, 110.0, 100.0, 100.0],
    )
    plugin_path = write_intraday_plugin(
        tmp_path,
        """
def generate_signal(ctx):
    if ctx.current_ts.minute == 30:
        return 1.0
    if ctx.current_ts.minute == 35:
        return -1.0
    return 0.0
""".lstrip(),
    )

    result = run_intraday_event_backtest(make_config(tmp_path, dataset_path, plugin_path))

    assert result.metrics["event_count"] == 3
    assert result.metrics["trade_count"] == 2
    assert result.metrics["long_trade_count"] == 1
    assert result.metrics["short_trade_count"] == 1
    assert result.metrics["exposure"] == pytest.approx(2.0 / 3.0)
    assert result.metrics["long_exposure"] == pytest.approx(1.0 / 3.0)
    assert result.metrics["short_exposure"] == pytest.approx(1.0 / 3.0)
    assert result.metrics["turnover"] == pytest.approx(4.0)


def test_cli_run_intraday_event_script_smoke_writes_outputs(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")
    output_root = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run-intraday-event-script",
            "--dataset-path",
            str(dataset_path),
            "--strategy-file",
            str(plugin_path),
            "--no-require-quality-approval",
            "--allow-legacy-bar-timestamps",
            "--from-date",
            "2024-01-02",
            "--schedule",
            "every-bar",
            "--execution-timing",
            "signal_close_to_same_close",
            "--allow-noncausal-same-close",
            "--end-of-data-position-policy",
            "mark",
            "--output-root",
            str(output_root),
            "--run-id",
            "cli-event",
        ],
    )

    output_dir = output_root / "cli-event"
    assert result.exit_code == 0, result.stdout
    assert "event_count=1" in result.stdout
    assert "strategy_call_count=1" in result.stdout
    assert (output_dir / "events.parquet").exists()
    assert (output_dir / "equity.parquet").exists()
    assert (output_dir / "trades.parquet").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["event_count"] == 1
