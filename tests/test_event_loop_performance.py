from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from time import perf_counter

import polars as pl
import pytest
from test_intraday_event_support import make_config

from backtest_engine.intraday.engine import validate_intraday_dataset
from backtest_engine.intraday.event.config import (
    IntradayEventExecutionConfig,
    IntradayEventSessionConfig,
)
from backtest_engine.intraday.event.loop import build_intraday_event_events
from backtest_engine.intraday.event.output_buffer import EventFrameBuffer
from backtest_engine.intraday.event.row_source import FrameRows
from backtest_engine.intraday.event.schedule import build_call_schedule


def test_event_output_is_flushed_to_columnar_chunks() -> None:
    buffer = EventFrameBuffer({"value": pl.Int64}, chunk_size=10)
    for value in range(25):
        buffer.append({"value": value})
        assert buffer.pending_row_count < 10

    assert buffer.finish().get_column("value").to_list() == list(range(25))


@pytest.mark.parametrize("row_count", [500_000, 5_000_000])
def test_large_every_bar_schedule_has_constant_python_memory(row_count: int) -> None:
    rows = FrameRows(pl.DataFrame({"value": pl.arange(0, row_count, eager=True)}))
    execution = IntradayEventExecutionConfig(
        timing="signal_close_to_same_close",
        allow_noncausal_same_close=True,
        end_of_data_position_policy="mark",
        from_date=date(2024, 1, 1),
        schedule="every-bar",
    )

    schedule = build_call_schedule(rows, execution, IntradayEventSessionConfig(), 1)
    for index in range(0, row_count, max(1, row_count // 100)):
        rows[index]

    assert schedule.every_bar is True
    assert schedule.calls == {}
    assert rows.cached_row_count <= 32


def test_event_loop_throughput_regression(tmp_path: Path) -> None:
    assert_event_loop_throughput(tmp_path, 20_000)


@pytest.mark.performance
@pytest.mark.skipif(
    os.environ.get("RUN_PERFORMANCE_TESTS") != "1",
    reason="500k-bar performance regression is enabled in the performance CI job",
)
def test_500k_event_loop_throughput_regression(tmp_path: Path) -> None:
    assert_event_loop_throughput(tmp_path, 500_000)


@pytest.mark.performance
@pytest.mark.skipif(
    os.environ.get("RUN_LARGE_PERFORMANCE_TESTS") != "1",
    reason="5m-bar performance regression is an explicit capacity check",
)
def test_5m_event_loop_throughput_regression(tmp_path: Path) -> None:
    assert_event_loop_throughput(tmp_path, 5_000_000)


def assert_event_loop_throughput(tmp_path: Path, row_count: int) -> None:
    frame = synthetic_event_frame(row_count)
    dataset_path = tmp_path / f"synthetic-{row_count}.parquet"
    plugin = tmp_path / "flat.py"
    plugin.write_text("def generate_signal(ctx, params): return 0\n", encoding="utf-8")
    config = make_config(
        tmp_path,
        dataset_path,
        plugin,
        execution_timing="signal_close_to_same_close",
    )
    validation = validate_intraday_dataset(frame, dataset_path)

    started = perf_counter()
    events = build_intraday_event_events(config, frame, validation)
    rows_per_second = events.height / max(perf_counter() - started, 1e-9)

    minimum_rate = float(os.environ.get("MIN_EVENT_ROWS_PER_SECOND", "500"))
    assert events.height == row_count - 1
    assert rows_per_second >= minimum_rate
    print(f"Flat event loop: {events.height} bars, {rows_per_second:.0f} bars/second")


def synthetic_event_frame(row_count: int) -> pl.DataFrame:
    index = pl.int_range(0, row_count, eager=True).alias("_index")
    price = (100.0 + pl.col("_index") % 10).cast(pl.Float64)
    return pl.DataFrame({"_index": index}).select(
        pl.lit("synthetic").alias("source"),
        pl.lit("CRYPTO").alias("exchange"),
        pl.lit("spot").alias("engine"),
        pl.lit("crypto").alias("market"),
        pl.lit("SPOT").alias("board"),
        pl.lit("BTCUSDT").alias("symbol"),
        pl.lit("BTCUSDT").alias("continuous_symbol"),
        pl.lit("1m").alias("timeframe"),
        (pl.lit(datetime(2024, 1, 2)) + pl.duration(minutes=pl.col("_index"))).alias("ts"),
        price.alias("open"),
        price.alias("high"),
        price.alias("low"),
        price.alias("close"),
        pl.lit(1.0).alias("volume"),
        price.alias("value"),
        pl.lit("USDT").alias("currency"),
        pl.lit("BTCUSDT").alias("active_contract"),
        pl.lit("BTCUSDT").alias("source_contract_symbol"),
        pl.lit("nearest_contract").alias("dataset_type"),
        pl.lit("unadjusted").alias("adjustment_mode"),
    )
