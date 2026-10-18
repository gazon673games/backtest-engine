"""Publishes a bounded live preview while an isolated backtest worker runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from backtest_engine.workbench.progress import write_progress

MAX_EQUITY_POINTS = 600
MAX_DECISIONS = 600
MIN_WRITE_INTERVAL_SECONDS = 0.15


class RunProgressReporter:
    def __init__(self, output_root: Path, run_id: str) -> None:
        self.output_root = output_root
        self.run_id = run_id
        self.equity: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.last_write = 0.0

    def phase(self, *, pct: float, stage: str, message: str) -> None:
        write_progress(
            self.output_root,
            self.run_id,
            status="running",
            stage=stage,
            message=message,
            pct=pct,
            extra={
                "job_type": "run",
                "live": {"equity": self.equity, "decisions": self.decisions},
            },
        )

    def __call__(
        self,
        processed: int,
        total: int,
        event_row: dict[str, Any] | None,
    ) -> None:
        if event_row is not None:
            append_bounded(self.equity, equity_point(event_row), MAX_EQUITY_POINTS)
            if is_decision(event_row):
                append_bounded(
                    self.decisions,
                    decision_point(event_row),
                    MAX_DECISIONS,
                )
        now = time.monotonic()
        if processed < total and now - self.last_write < MIN_WRITE_INTERVAL_SECONDS:
            return
        self.last_write = now
        ratio = 1.0 if total <= 0 else min(1.0, processed / total)
        write_progress(
            self.output_root,
            self.run_id,
            status="running",
            stage="backtest",
            message=f"processed {processed} of {total} strategy bars",
            pct=0.1 + (0.72 * ratio),
            extra={
                "job_type": "run",
                "processed_bars": processed,
                "total_bars": total,
                "live": {"equity": self.equity, "decisions": self.decisions},
            },
        )


def equity_point(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts": iso_value(row.get("next_ts") or row["ts"]),
        "equity": row["equity"],
        "equity_cash": row["equity_cash"],
    }


def decision_point(row: dict[str, Any]) -> dict[str, Any]:
    filled_delta = row.get("filled_contract_delta")
    if filled_delta is None:
        filled_delta = row["contract_delta"]
    return {
        "ts": iso_value(row["ts"]),
        "fill_ts": iso_value(row.get("fill_ts") or row["ts"]),
        "target_position": row["target_position"],
        "target_contracts": row["target_contracts"],
        "contract_delta": row["contract_delta"],
        "filled_contract_delta": filled_delta,
        "order_filled": row.get("order_filled", float(filled_delta or 0.0) != 0.0),
        "reason": row["strategy_reason"] or row["signal_reason"],
    }


def is_decision(row: dict[str, Any]) -> bool:
    return bool(
        row["strategy_called"]
        or row["missing_decision"]
        or float(row["contract_delta"] or 0.0) != 0.0
    )


def append_bounded(items: list[dict[str, Any]], item: dict[str, Any], maximum: int) -> None:
    if items and items[-1].get("ts") == item.get("ts"):
        items[-1] = item
    else:
        items.append(item)
    if len(items) > maximum:
        last = items[-1]
        sampled = items[::2]
        if sampled[-1] is not last:
            sampled.append(last)
        items[:] = sampled


def iso_value(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)
