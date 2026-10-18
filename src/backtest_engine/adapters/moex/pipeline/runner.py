"""Orchestrates normalization, aggregation, roll construction, audit, and approval."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from backtest_engine.adapters.moex.pipeline.models import (
    MoexDataPipelineConfig,
    MoexDataPipelineResult,
)
from backtest_engine.adapters.moex.pipeline.raw_storage import write_raw_snapshot
from backtest_engine.adapters.moex.pipeline.stages import (
    aggregate_bars,
    approval_payload,
    build_continuous_nearest,
    build_nearest_contract,
    gap_audit,
    normalize_moex_candles,
    repair_gaps,
    roll_audit,
)
from backtest_engine.adapters.proxy import open_adapter_url
from backtest_engine.adapters.raw_json import read_json as read_raw_json
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.bar_availability import (
    ensure_bar_availability,
    finalized_as_of,
    source_as_of_from_payload,
)
from backtest_engine.data.store import file_sha256


def run_moex_data_pipeline(config: MoexDataPipelineConfig) -> MoexDataPipelineResult:
    version = uuid4().hex
    config = replace(
        config,
        output_root=config.output_root / "versions" / version,
        dataset_id=f"{config.dataset_id}:{version}",
    )
    output_root = config.output_root
    raw_dir = output_root / "raw"
    normalized_dir = output_root / "normalized"
    bars_dir = output_root / "bars"
    reports_dir = output_root / "reports"
    for path in (raw_dir, normalized_dir, bars_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    raw_frame, raw_payload = load_raw_moex_candles(config)
    source_as_of = config.source_as_of or source_as_of_from_payload(raw_payload)
    raw_snapshot_path = write_raw_snapshot(raw_dir, raw_payload)
    normalized = normalize_moex_candles(
        raw_frame,
        continuous_symbol=config.continuous_symbol,
        timeframe=config.base_timeframe,
    )
    normalized, base_finality = finalized_as_of(
        normalized,
        timeframe=config.base_timeframe,
        source_as_of=source_as_of,
        source_delay_minutes=config.source_delay_minutes,
        timestamp_timezone="Europe/Moscow",
    )
    repaired, repair_result = repair_gaps(
        normalized,
        timeframe=config.base_timeframe,
        policy=config.gap_repair_policy,
        session_windows=config.session_windows,
    )
    repaired = ensure_bar_availability(repaired, config.base_timeframe)
    normalized_path = normalized_dir / "candles.parquet"
    repaired.write_parquet(normalized_path)

    aggregate_frames = build_aggregate_frames(
        repaired,
        config.aggregate_timeframes,
        source_as_of=source_as_of,
        source_delay_minutes=config.source_delay_minutes,
    )
    aggregate_paths = write_aggregates(aggregate_frames, bars_dir)
    research_frame = research_aggregate_frame(repaired, aggregate_frames)
    nearest = build_nearest_contract(research_frame)
    continuous = build_continuous_nearest(research_frame)
    nearest_path = bars_dir / "nearest.parquet"
    continuous_path = bars_dir / "continuous_nearest.parquet"
    nearest.write_parquet(nearest_path)
    continuous.write_parquet(continuous_path)

    gap_result = gap_audit(
        research_frame,
        research_frame["timeframe"][0],
        config.session_windows,
    )
    roll_result = roll_audit(continuous, max_gap_pct=config.max_roll_gap_pct)
    if isinstance(raw_payload, dict) and raw_payload.get("chain_audit"):
        roll_result["selection"] = raw_payload["chain_audit"]
    approval = approval_payload(
        gap_result=gap_result,
        roll_result=roll_result,
        repair_result=repair_result,
        gap_policy=config.gap_policy,
        roll_gap_policy=config.roll_gap_policy,
        max_gap_ratio=config.max_gap_ratio,
        max_roll_gap_pct=config.max_roll_gap_pct,
    )
    approval["finality"] = base_finality
    approval["dataset"] = {
        "dataset_id": config.dataset_id,
        "path": str(nearest_path),
        "sha256": file_sha256(nearest_path),
        "continuous_symbol": config.continuous_symbol,
        "timeframe": str(research_frame["timeframe"][0]),
        "row_count": nearest.height,
    }
    approval["session_model"] = {
        "session_aware": True,
        "windows": [
            {"id": name, "start": start, "end": end} for name, start, end in config.session_windows
        ],
    }
    gap_audit_path = reports_dir / "gap_audit.json"
    gap_repair_path = reports_dir / "gap_repair.json"
    roll_audit_path = reports_dir / "roll_audit.json"
    approval_path = reports_dir / "approval.json"
    manifest_path = reports_dir / "pipeline_manifest.json"
    write_json(gap_audit_path, gap_result)
    write_json(gap_repair_path, repair_result)
    write_json(roll_audit_path, roll_result)
    write_json(approval_path, approval)
    write_json(
        manifest_path,
        pipeline_manifest(config, raw_snapshot_path, approval_path, base_finality),
    )

    return MoexDataPipelineResult(
        output_root=output_root,
        dataset_id=config.dataset_id,
        raw_snapshot_path=raw_snapshot_path,
        normalized_path=normalized_path,
        nearest_path=nearest_path,
        continuous_path=continuous_path,
        manifest_path=manifest_path,
        approval_path=approval_path,
        gap_audit_path=gap_audit_path,
        gap_repair_path=gap_repair_path,
        roll_audit_path=roll_audit_path,
        aggregate_paths=aggregate_paths,
        approval_status=str(approval["approval_status"]),
        source_as_of=str(base_finality["source_as_of"]),
        finalized_through=str(base_finality["finalized_through"]),
    )


def load_raw_moex_candles(config: MoexDataPipelineConfig) -> tuple[pl.DataFrame, Any]:
    if config.raw_snapshot_path is not None:
        path = config.raw_snapshot_path
        if not path.exists():
            raise DatasetValidationError(f"Raw MOEX snapshot does not exist: {path}")
        if path.suffix.lower() == ".parquet":
            frame = pl.read_parquet(path)
            return frame, {"source_path": str(path), "rows": frame.to_dicts()}
        if path.suffix.lower() == ".csv":
            frame = pl.read_csv(path, try_parse_dates=True)
            return frame, {"source_path": str(path), "rows": frame.to_dicts()}
        payload = read_raw_json(path)
        return frame_from_payload(payload), payload
    if config.source_url is None:
        raise DatasetValidationError("raw_snapshot_path or source_url is required")
    with open_adapter_url("moex", config.source_url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return frame_from_payload(payload), payload


def frame_from_payload(payload: Any) -> pl.DataFrame:
    if isinstance(payload, list):
        return pl.DataFrame(payload)
    if not isinstance(payload, dict):
        raise DatasetValidationError("MOEX raw JSON payload must be object or list")
    if "rows" in payload:
        return pl.DataFrame(payload["rows"])
    candles = payload.get("candles") or payload.get("securities")
    if isinstance(candles, dict) and "columns" in candles and "data" in candles:
        return pl.DataFrame(
            [dict(zip(candles["columns"], row, strict=False)) for row in candles["data"]]
        )
    if isinstance(candles, list):
        return pl.DataFrame(candles)
    raise DatasetValidationError("MOEX raw JSON payload does not contain rows/candles")


def build_aggregate_frames(
    normalized: pl.DataFrame,
    timeframes: list[str],
    *,
    source_as_of: datetime,
    source_delay_minutes: int,
) -> dict[str, pl.DataFrame]:
    frames = {}
    for timeframe in timeframes:
        frame = aggregate_bars(normalized, timeframe)
        frames[timeframe] = finalized_as_of(
            frame,
            timeframe=timeframe,
            source_as_of=source_as_of,
            source_delay_minutes=source_delay_minutes,
            timestamp_timezone="Europe/Moscow",
        )[0]
    return frames


def write_aggregates(frames: dict[str, pl.DataFrame], bars_dir: Path) -> dict[str, Path]:
    paths = {}
    for timeframe, frame in frames.items():
        path = bars_dir / f"bars_{timeframe}.parquet"
        frame.write_parquet(path)
        paths[timeframe] = path
    return paths


def research_aggregate_frame(
    normalized: pl.DataFrame,
    aggregate_frames: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    return next(iter(aggregate_frames.values()), normalized)


def pipeline_manifest(
    config: MoexDataPipelineConfig,
    raw_snapshot_path: Path,
    approval_path: Path,
    finality: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_id": config.dataset_id,
        "continuous_symbol": config.continuous_symbol,
        "base_timeframe": config.base_timeframe,
        "aggregate_timeframes": config.aggregate_timeframes,
        "gap_policy": config.gap_policy,
        "gap_repair_policy": config.gap_repair_policy,
        "roll_gap_policy": config.roll_gap_policy,
        "max_gap_ratio": config.max_gap_ratio,
        "max_roll_gap_pct": config.max_roll_gap_pct,
        "bar_time_semantics": finality["bar_time_semantics"],
        "source_as_of": finality["source_as_of"],
        "source_delay_minutes": finality["source_delay_minutes"],
        "finalized_through": finality["finalized_through"],
        "session_windows": [list(window) for window in config.session_windows],
        "raw_snapshot_path": str(raw_snapshot_path),
        "raw_snapshot_sha256": file_sha256(raw_snapshot_path),
        "approval_path": str(approval_path),
        "approval_sha256": file_sha256(approval_path),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
