# ruff: noqa
from test_intraday_event_support import *  # noqa: F401,F403


def test_invalid_event_target_is_rejected(tmp_path: Path) -> None:
    dataset_path = write_intraday_dataset(tmp_path)
    plugin_path = write_intraday_plugin(tmp_path, "def generate_signal(ctx):\n    return 2.0\n")

    with pytest.raises(StrategyPluginError, match="unsupported signal value"):
        run_intraday_event_backtest(make_config(tmp_path, dataset_path, plugin_path))
