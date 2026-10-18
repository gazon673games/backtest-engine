from __future__ import annotations

import pytest
from test_intraday_event_workflow_support import write_intraday_dataset, write_mode_plugin

from backtest_engine.intraday.event.batch.config import IntradayEventBatchConfig
from backtest_engine.intraday.event.batch.runner import run_intraday_event_batch


def batch_config(tmp_path, *, modes: list[str]) -> IntradayEventBatchConfig:
    return IntradayEventBatchConfig(
        experiment_id="resume",
        output_root=tmp_path / "batch",
        dataset_paths=[write_intraday_dataset(tmp_path)],
        quality_approval_required=False,
        allow_legacy_bar_timestamps=True,
        strategy_file=write_mode_plugin(tmp_path),
        timing="signal_close_to_same_close",
        allow_noncausal_same_close=True,
        end_of_data_position_policy="mark",
        schedule="times",
        decision_times=[],
        grid={"decision_time": ["13:30"], "mode": modes},
        max_workers=1,
    )


def test_completed_run_is_reused_only_when_output_hashes_match(tmp_path) -> None:
    config = batch_config(tmp_path, modes=["long"])

    first = run_intraday_event_batch(config)
    second = run_intraday_event_batch(config)

    assert first.resumed_run_count == 0
    assert second.resumed_run_count == 1
    assert second.summary_rows[0]["cache_status"] == "hit"

    events_path = second.output_dir / "runs" / second.summary_rows[0]["run_id"] / "events.parquet"
    events_path.write_bytes(b"corrupt")
    repaired = run_intraday_event_batch(config)

    assert repaired.resumed_run_count == 0
    assert repaired.summary_rows[0]["cache_status"] == "miss"


def test_duplicate_grid_cases_are_skipped_before_execution(tmp_path) -> None:
    result = run_intraday_event_batch(batch_config(tmp_path, modes=["long", "long"]))

    assert result.successful_run_count == 1
    assert result.skipped_duplicate_count == 1


def test_effective_capital_has_a_distinct_stable_output_identity(tmp_path):
    config = batch_config(tmp_path, modes=["long"])
    first = run_intraday_event_batch(config)
    config.portfolio.initial_capital *= 2
    second = run_intraday_event_batch(config)
    assert first.summary_rows[0]["run_path"] != second.summary_rows[0]["run_path"]
    repeated = run_intraday_event_batch(config)
    assert repeated.resumed_run_count == 1
    assert repeated.summary_rows[0]["run_path"] == second.summary_rows[0]["run_path"]


@pytest.mark.parametrize("continue_on_error", [False, True])
def test_batch_stops_before_next_case_when_errors_are_fatal(
    tmp_path, monkeypatch, continue_on_error: bool
) -> None:
    calls = []

    def failing_case(*args):
        calls.append(args[-1]["mode"])
        raise RuntimeError("case failed")

    monkeypatch.setattr("backtest_engine.intraday.event.batch.runner.run_batch_case", failing_case)
    config = batch_config(tmp_path, modes=["long", "flat"])
    config.continue_on_error = continue_on_error

    result = run_intraday_event_batch(config)

    assert calls == (["long", "flat"] if continue_on_error else ["long"])
    assert result.successful_run_count == 0
    assert result.failed_run_count == len(calls)
