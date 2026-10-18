from pathlib import Path

from test_intraday_event_support import make_config, write_intraday_dataset, write_intraday_plugin

from backtest_engine.intraday.event.engine import run_intraday_event_backtest


def test_event_outputs_default_to_parquet_only(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 1.0\n")

    result = run_intraday_event_backtest(make_config(tmp_path, dataset_path, plugin_path))

    assert (result.output_dir / "events.parquet").exists()
    assert not (result.output_dir / "events.csv").exists()
