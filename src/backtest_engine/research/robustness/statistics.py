"""Daily block resampling and family-only multiple-testing diagnostics."""

import polars as pl

from backtest_engine.research.selection_bias import analyze_selection_bias
from backtest_engine.research.validation.analytics import block_bootstrap_metrics
from backtest_engine.research.validation.config import ResearchValidationConfig

from .models import Protocol


def select_on_training(rows: list[dict], group: str) -> dict:
    candidates = [
        r
        for r in rows
        if r["group"] == group
        and r.get("status") == "completed"
        and r.get("train_total_return") is not None
        and r.get("train_fill_rows", 0) > 0
    ]
    if not candidates:
        raise ValueError(f"no active training candidates in family {group}")
    # Check-period returns never participate, including in tie-breaking.
    return min(candidates, key=lambda r: (-r["train_total_return"], r["run_id"]))


def family_diagnostics(protocol: Protocol, rows: list[dict], daily: dict[str, pl.DataFrame]):
    families, bias_rows, deflated_rows = [], [], []
    for group in protocol.family_groups:
        members = [r for r in rows if r["group"] == group]
        valid = [r for r in members if r["status"] == "completed"]
        totals = [r["total_return"] for r in valid]
        checks = [r["check_total_return"] for r in valid if r["check_total_return"] is not None]
        selected = None
        if len(valid) == len(members) and any(r.get("train_fill_rows", 0) for r in valid):
            selected = select_on_training(valid, group)["run_id"]
        families.append(
            {
                "group": group,
                "planned_count": len(members),
                "completed_count": len(valid),
                "failed_count": len(members) - len(valid),
                "inactive_count": sum(not r["active"] for r in valid),
                "positive_all_count": sum(v > 0 for v in totals),
                "check_count": len(checks),
                "positive_check_count": sum(v > 0 for v in checks),
                "all_min": min(totals) if totals else None,
                "all_median": pl.Series(totals).median() if totals else None,
                "all_max": max(totals) if totals else None,
                "check_min": min(checks) if checks else None,
                "check_median": pl.Series(checks).median() if checks else None,
                "check_max": max(checks) if checks else None,
                "selected_on_train": selected,
            }
        )
        if len(valid) != len(members):
            bias_rows.append({"group": group, "status": "skipped_incomplete_family"})
            continue
        if len(valid) < 2 or not protocol.selection_bias_iterations:
            continue
        frames = [daily[r["run_id"]] for r in valid]
        if any(not f["date"].equals(frames[0]["date"]) for f in frames[1:]):
            bias_rows.append({"group": group, "status": "skipped_different_daily_coverage"})
            continue
        analysis = analyze_selection_bias(
            [
                (
                    r,
                    daily[r["run_id"]]
                    .select("ts", "net_return")
                    .with_columns(pl.col("ts").dt.date()),
                )
                for r in valid
            ],
            bootstrap_iterations=protocol.selection_bias_iterations,
            seed=protocol.seed,
        )
        bias_rows.extend(
            {"group": group, "status": "computed", **r} for r in analysis.summary.to_dicts()
        )
        deflated_rows.extend({"group": group, **r} for r in analysis.deflated_sharpe.to_dicts())
    return families, bias_rows, deflated_rows


def baseline_bootstrap(protocol: Protocol, daily: dict[str, pl.DataFrame]) -> list[dict]:
    if protocol.baseline not in daily:
        return []
    frame = daily[protocol.baseline]
    rows = []
    for segment, subset in (
        ("all", frame),
        ("check", frame.filter(pl.col("date") >= protocol.split_date)),
    ):
        for block in protocol.bootstrap_blocks_days:
            if subset.height < 2 * block:
                rows.append(
                    {
                        "segment": segment,
                        "requested_block_days": block,
                        "status": "skipped_insufficient_days",
                    }
                )
                continue
            config = ResearchValidationConfig(
                experiment_path=protocol.output_root,
                block_bootstrap_length=block,
                block_bootstrap_iterations=protocol.bootstrap_iterations,
                bootstrap_seed=protocol.seed + block,
            )
            result = block_bootstrap_metrics({"run_id": protocol.baseline}, subset, config, 0)
            if result is not None:
                rows.append(
                    {
                        "segment": segment,
                        "requested_block_days": block,
                        "status": "computed",
                        **result,
                    }
                )
    return rows
