# ruff: noqa
from test_platform_layers_support import *  # noqa: F401,F403


def test_golden_intraday_event_reversal_pnl_and_trades(tmp_path: Path) -> None:
    dataset_path = write_dataset(
        tmp_path,
        closes=[100.0, 110.0, 90.0, 95.0],
        highs=[100.0, 110.0, 90.0, 95.0],
        lows=[100.0, 110.0, 90.0, 95.0],
    )
    plugin_path = write_plugin(
        tmp_path,
        """
def generate_signal(ctx):
    if ctx.current_ts.minute == 30:
        return {"target": 1.0, "reason": "open_long"}
    if ctx.current_ts.minute == 35:
        return {"target": -1.0, "reason": "reverse_short"}
    return {"target": 0.0, "reason": "flat"}
""".lstrip(),
    )

    result = run_intraday_event_backtest(
        make_config(
            tmp_path,
            dataset_path,
            plugin_path,
            sizing_mode="fixed_contracts",
            fixed_contracts=1.0,
            initial_capital=1000.0,
        )
    )
    events = pl.read_parquet(result.output_dir / "events.parquet")
    trades = pl.read_parquet(result.output_dir / "trades.parquet")

    assert events.get_column("target_position").to_list() == [1.0, -1.0, 0.0]
    assert events.get_column("gross_pnl").to_list() == [10.0, 20.0, 0.0]
    assert result.metrics["final_equity_cash"] == pytest.approx(1030.0)
    assert result.metrics["total_return"] == pytest.approx(0.03)
    assert trades.get_column("direction").to_list() == ["long", "short"]
    assert trades.get_column("gross_pnl").to_list() == [10.0, 20.0]


def test_limit_order_unfilled_keeps_previous_position(tmp_path: Path) -> None:
    dataset_path = write_dataset(
        tmp_path,
        closes=[100.0, 110.0, 120.0],
        highs=[100.0, 112.0, 120.0],
        lows=[100.0, 105.0, 120.0],
    )
    plugin_path = write_plugin(
        tmp_path,
        """
def generate_signal(ctx):
    return {
        "target": 1.0,
        "reason": "patient_buy",
        "order": {"type": "limit", "limit_price": 99.0},
    }
""".lstrip(),
    )

    result = run_intraday_event_backtest(make_config(tmp_path, dataset_path, plugin_path))
    events = pl.read_parquet(result.output_dir / "events.parquet")
    first = events.row(0, named=True)

    assert first["requested_position"] == 1.0
    assert first["target_position"] == 0.0
    assert first["order_type"] == "limit"
    assert first["order_status"] == "unfilled"
    assert first["signal_reason"] == "order_unfilled"
    assert result.metrics["unfilled_order_count"] == 2


def test_intraday_event_cli_can_resolve_dataset_from_catalog(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    plugin_path = write_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_version": 1,
                "entries": [
                    {
                        "dataset_id": "fixture:5m:unadjusted",
                        "dataset_path": str(dataset_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "run-intraday-event-script",
            "--catalog-path",
            str(catalog_path),
            "--dataset-id",
            "fixture:5m:unadjusted",
            "--strategy-file",
            str(plugin_path),
            "--no-require-quality-approval",
            "--allow-legacy-bar-timestamps",
            "--end-of-data-position-policy",
            "mark",
            "--output-root",
            str(tmp_path / "out"),
            "--run-id",
            "catalog_run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "catalog_run" / "events.parquet").exists()


def test_strict_catalog_enforcement_requires_bound_artifacts(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    plugin_path = write_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")
    catalog_path = write_catalog_bundle(tmp_path, dataset_path)

    result = run_intraday_event_backtest(
        IntradayEventBacktestConfig(
            dataset=IntradayEventDatasetConfig(
                path=dataset_path,
                catalog_path=catalog_path,
                dataset_id="fixture:5m:unadjusted",
                catalog_enforcement="strict",
                allow_legacy_bar_timestamps=True,
            ),
            strategy=IntradayEventStrategyConfig(strategy_file=plugin_path),
            execution=IntradayEventExecutionConfig(
                from_date=date(2024, 1, 2),
                end_of_data_position_policy="mark",
            ),
            output=IntradayEventOutputConfig(root=tmp_path / "out", run_id="strict"),
        )
    )

    catalog_approval = json.loads(
        (result.output_dir / "catalog_approval.json").read_text(encoding="utf-8")
    )
    assert catalog_approval["status"] == "approved"
    assert (result.output_dir / "experiment_manifest.json").exists()


def test_portfolio_combine_aggregates_two_intraday_runs(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    long_plugin = write_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n", "long.py")
    short_plugin = write_plugin(
        tmp_path, "def generate_signal(ctx):\n    return -1.0\n", "short.py"
    )
    long_result = run_intraday_event_backtest(
        make_config(tmp_path, dataset_path, long_plugin, run_id="long")
    )
    short_result = run_intraday_event_backtest(
        make_config(tmp_path, dataset_path, short_plugin, run_id="short")
    )

    result = combine_intraday_event_portfolio(
        PortfolioCombineConfig(
            run_paths=[long_result.output_dir, short_result.output_dir],
            output_path=tmp_path / "portfolio",
        )
    )
    equity = pl.read_parquet(result.output_path / "portfolio_equity.parquet")

    assert result.run_count == 2
    assert equity.height == 2
    assert result.final_equity_cash == pytest.approx(2.018181818181818)
    assert result.total_return == pytest.approx(0.009090909090909038)


def test_research_validation_report_writes_oos_and_robustness_tables(tmp_path: Path) -> None:
    dataset_path = write_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 3, 13, 30),
            datetime(2024, 1, 3, 13, 35),
            datetime(2024, 1, 4, 13, 30),
            datetime(2024, 1, 4, 13, 35),
        ],
        closes=[100.0, 101.0, 101.0, 103.0, 103.0, 106.0],
    )
    plugin_path = write_plugin(
        tmp_path,
        """
def generate_signal(ctx, params=None):
    return 1.0 if params.get("mode") == "long" else 0.0
""".lstrip(),
    )
    batch = run_intraday_event_batch(
        IntradayEventBatchConfig(
            experiment_id="research",
            output_root=tmp_path / "batch",
            dataset_paths=[dataset_path],
            quality_approval_required=False,
            allow_legacy_bar_timestamps=True,
            strategy_file=plugin_path,
            end_of_data_position_policy="mark",
            grid={"mode": ["long", "flat"], "decision_time": ["13:30"]},
            schedule="times",
        )
    )

    result = generate_research_validation_report(
        ResearchValidationConfig(
            experiment_path=batch.output_dir,
            split_date=date(2024, 1, 4),
        )
    )

    assert result.run_count == 2
    assert result.walk_forward_path.exists()
    assert result.robustness_path.exists()
    assert result.guardrails_path.exists()
    assert result.bootstrap_path.exists()
    assert result.block_bootstrap_path.exists()
    assert result.cost_stress_path.exists()
    assert result.regime_path.exists()
    assert result.report_path.exists()
    report_text = result.report_path.read_text(encoding="utf-8")
    assert "Optimization Guardrails" in report_text
    assert "Block Bootstrap" in report_text
    assert "Cost Stress" in report_text


def test_intraday_event_research_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    dataset_path = write_dataset(
        tmp_path,
        timestamps=[
            datetime(2024, 1, 2, 13, 30),
            datetime(2024, 1, 2, 13, 35),
            datetime(2024, 1, 3, 13, 30),
            datetime(2024, 1, 3, 13, 35),
            datetime(2024, 1, 4, 13, 30),
            datetime(2024, 1, 4, 13, 35),
        ],
        closes=[100.0, 101.0, 101.0, 102.0, 102.0, 103.0],
    )
    catalog_path = write_catalog_bundle(tmp_path, dataset_path)
    plugin_path = write_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        f"""
experiment_id = "pipeline"
output_root = "{(tmp_path / "batch").as_posix()}"
catalog_path = "{catalog_path.as_posix()}"
dataset_ids = ["fixture:5m:unadjusted"]
strategy_file = "{plugin_path.as_posix()}"
from_date = "2024-01-02"
allow_legacy_bar_timestamps = true
end_of_data_position_policy = "mark"
schedule = "times"
decision_tolerance_minutes = 5

[grid]
decision_time = ["13:30"]
""".lstrip(),
        encoding="utf-8",
    )

    result = run_intraday_event_research_pipeline(
        IntradayEventResearchPipelineConfig(
            config_path=config_path,
            catalog_enforcement="strict",
            report_charts=False,
            split_date=date(2024, 1, 4),
        )
    )

    assert result.batch.successful_run_count == 1
    assert result.manifest_path.exists()
    assert result.research_validation.report_path.exists()
