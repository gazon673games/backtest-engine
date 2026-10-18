"""Implements deterministic frame transformations used by the MOEX data pipeline."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl
from ohlcv_data_store.market_values import market_value_issues

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.timeframes import parse_timeframe_minutes
from backtest_engine.data.bar_availability import ensure_bar_availability
from backtest_engine.data.session_gap_audit import SessionWindows, same_session_window
from backtest_engine.data.session_gap_audit import session_gap_audit as gap_audit

from .quality import approval_payload as approval_payload
from .quality import roll_audit as roll_audit

COLUMN_ALIASES = {
    "ts": ("ts", "begin", "begin_time", "TRADEDATE", "tradedate"),
    "open": ("open", "OPEN", "open_price"),
    "high": ("high", "HIGH"),
    "low": ("low", "LOW"),
    "close": ("close", "CLOSE", "close_price"),
    "volume": ("volume", "VOLUME", "vol"),
    "value": ("value", "VALUE", "turnover"),
    "symbol": ("symbol", "SECID", "secid"),
    "active_contract": ("active_contract", "SECID", "secid", "symbol"),
    "source": ("source",),
    "engine": ("engine",),
    "market": ("market",),
    "board": ("board", "BOARDID", "boardid"),
    "currency": ("currency", "CURRENCYID", "currencyid"),
}


def normalize_moex_candles(
    frame: pl.DataFrame,
    *,
    continuous_symbol: str,
    timeframe: str,
) -> pl.DataFrame:
    if frame.height == 0:
        raise DatasetValidationError("MOEX raw candle frame is empty")
    columns = {target: _source_column(frame, aliases) for target, aliases in COLUMN_ALIASES.items()}
    missing = [name for name in ("ts", "open", "high", "low", "close") if columns[name] is None]
    if missing:
        raise DatasetValidationError(f"MOEX raw candles missing columns: {', '.join(missing)}")
    normalized = frame.select(
        [
            _timestamp_expr(frame, columns["ts"]),
            pl.col(columns["open"]).cast(pl.Float64).alias("open"),
            pl.col(columns["high"]).cast(pl.Float64).alias("high"),
            pl.col(columns["low"]).cast(pl.Float64).alias("low"),
            pl.col(columns["close"]).cast(pl.Float64).alias("close"),
            _optional_float(columns["volume"], "volume"),
            _optional_float(columns["value"], "value"),
            _optional_string(columns["symbol"], "symbol", continuous_symbol),
            _optional_string(columns["active_contract"], "active_contract", None),
            _optional_string(columns["source"], "source", "moex_iss"),
            _optional_string(columns["engine"], "engine", "futures"),
            _optional_string(columns["market"], "market", "forts"),
            _optional_string(columns["board"], "board", "RFUD"),
            _optional_string(columns["currency"], "currency", "RUB"),
        ]
    )
    issues = market_value_issues(normalized)
    if issues:
        raise DatasetValidationError("MOEX candles: " + "; ".join(issues))
    enriched = normalized.with_columns(
        [
            pl.lit("MOEX").alias("exchange"),
            pl.lit("nearest_contract").alias("dataset_type"),
            pl.lit("unadjusted").alias("adjustment_mode"),
            pl.lit(continuous_symbol).alias("continuous_symbol"),
            pl.lit(timeframe).alias("timeframe"),
            pl.col("active_contract").fill_null(pl.col("symbol")).alias("source_contract_symbol"),
        ]
    ).sort("ts")
    return ensure_bar_availability(enriched, timeframe)


def aggregate_bars(frame: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    minutes = parse_timeframe_minutes(timeframe)
    aggregated = (
        frame.sort("ts")
        .group_by_dynamic("ts", every=f"{minutes}m")
        .agg(
            [
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                pl.col("value").sum(),
                pl.col("symbol").last(),
                pl.col("active_contract").last(),
                pl.col("source_contract_symbol").last(),
                pl.col("continuous_symbol").last(),
                pl.col("source").last(),
                pl.col("exchange").last(),
                pl.col("engine").last(),
                pl.col("market").last(),
                pl.col("board").last(),
                pl.col("currency").last(),
                pl.col("dataset_type").last(),
                pl.col("adjustment_mode").last(),
                pl.col("is_final").all(),
            ]
        )
        .with_columns(pl.lit(timeframe).alias("timeframe"))
    )
    return ensure_bar_availability(aggregated, timeframe)


def repair_gaps(
    frame: pl.DataFrame,
    *,
    timeframe: str,
    policy: str,
    session_windows: SessionWindows,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    allowed = {"none", "zero_volume_forward_fill"}
    if policy not in allowed:
        raise DatasetValidationError(f"Unknown gap repair policy: {policy}")
    if policy == "none":
        audit = gap_audit(frame, timeframe, session_windows)
        return frame, {
            "status": "not_required" if audit["gap_count"] == 0 else "not_applied",
            "policy": policy,
            "gap_count": audit["gap_count"],
            "synthetic_bar_count": 0,
            "repairs": [],
        }

    minutes = parse_timeframe_minutes(timeframe)
    rows = frame.sort("ts").to_dicts()
    repaired_rows: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row["is_synthetic"] = False
        row["repair_reason"] = None
        repaired_rows.append(row)
        if index >= len(rows) - 1:
            continue
        repairs.extend(
            _synthetic_gap_rows(
                row,
                rows[index + 1],
                minutes,
                repaired_rows,
                session_windows,
            )
        )
    repaired = pl.DataFrame(repaired_rows) if repaired_rows else frame
    return repaired.sort("ts"), {
        "status": "not_required" if not repairs else "repaired",
        "policy": policy,
        "gap_count": len(repairs),
        "synthetic_bar_count": sum(int(repair["synthetic_bar_count"]) for repair in repairs),
        "repairs": repairs[:100],
    }


def build_nearest_contract(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort("ts").with_columns(
        [
            pl.col("active_contract").fill_null(pl.col("source_contract_symbol")),
            pl.col("source_contract_symbol").fill_null(pl.col("active_contract")),
            pl.lit("nearest_contract").alias("dataset_type"),
        ]
    )


def build_continuous_nearest(frame: pl.DataFrame) -> pl.DataFrame:
    sorted_frame = build_nearest_contract(frame)
    contract = pl.col("active_contract")
    return sorted_frame.with_columns(
        [
            (contract != contract.shift(1)).fill_null(False).alias("roll_flag"),
            contract.alias("source_contract_symbol"),
        ]
    )


def _source_column(frame: pl.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower_map = {column.lower(): column for column in frame.columns}
    for alias in aliases:
        if alias in frame.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _synthetic_gap_rows(
    previous: dict[str, Any],
    current: dict[str, Any],
    minutes: int,
    repaired_rows: list[dict[str, Any]],
    session_windows: SessionWindows,
) -> list[dict[str, Any]]:
    if not same_session_window(previous["ts"], current["ts"], session_windows):
        return []
    delta_minutes = (current["ts"] - previous["ts"]).total_seconds() / 60.0
    if delta_minutes <= minutes * 1.5:
        return []
    synthetic_bar_count = 0
    next_ts = previous["ts"]
    interval = timedelta(minutes=minutes)
    previous_close = float(previous["close"])
    while True:
        next_ts += interval
        if next_ts >= current["ts"]:
            break
        synthetic = dict(previous)
        synthetic.update(
            {
                "ts": next_ts,
                "open": previous_close,
                "high": previous_close,
                "low": previous_close,
                "close": previous_close,
                "volume": 0.0,
                "value": 0.0,
                "is_synthetic": True,
                "repair_reason": "zero_volume_forward_fill",
            }
        )
        synthetic_bar_count += 1
        repaired_rows.append(synthetic)
    return [
        {
            "previous_ts": previous["ts"].isoformat(),
            "current_ts": current["ts"].isoformat(),
            "gap_minutes": delta_minutes,
            "synthetic_bar_count": synthetic_bar_count,
        }
    ]


def _timestamp_expr(frame: pl.DataFrame, column: str | None) -> pl.Expr:
    if column is None:
        raise DatasetValidationError("MOEX raw candles missing timestamp column")
    dtype = frame.schema[column]
    if dtype in (pl.String, pl.Utf8):
        return pl.col(column).str.to_datetime(strict=False).alias("ts")
    return pl.col(column).cast(pl.Datetime).alias("ts")


def _optional_float(column: str | None, alias: str) -> pl.Expr:
    if column is None:
        return pl.lit(0.0).alias(alias)
    return pl.col(column).cast(pl.Float64).alias(alias)


def _optional_string(
    column: str | None,
    alias: str,
    default: str | None,
) -> pl.Expr:
    if column is None:
        return pl.lit(default).alias(alias)
    return pl.col(column).cast(pl.String).alias(alias)
