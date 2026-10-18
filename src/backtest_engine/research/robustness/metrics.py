"""Analyze native portfolio NAV; verify execution and accounting before statistics."""

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.research.validation.stats import max_drawdown, segment_metrics

from .provenance import read_json


def audit_run(path: Path) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    config, native = read_json(path / "config.json"), read_json(path / "metrics.json")
    quality = read_json(path / "quality_approval.json")
    catalog = read_json(path / "catalog_approval.json")
    if (
        not config["dataset"]["quality_approval_required"]
        or quality["status"] != "approved"
        or config["dataset"]["catalog_enforcement"] != "strict"
        or catalog["status"] != "approved"
        or not catalog["approved_for_research"]
    ):
        raise ValueError(f"{path}: native quality/catalog approval required")
    if (
        config["execution"]["allow_noncausal_same_close"]
        or not config["strategy"]["strict_runtime"]
    ):
        raise ValueError(f"{path}: noncausal or permissive native runtime")
    ledger = pl.read_parquet(path / "portfolio_ledger.parquet")
    events = pl.read_parquet(path / "events.parquet")
    capital = float(config["portfolio"]["initial_capital"])
    tolerance = max(1e-8, capital * 1e-10)
    if not ledger.height or ledger.height != events.height:
        raise ValueError(f"{path}: empty or unmatched event/ledger rows")
    # Native ledger serializes aware instants as UTC-naive Datetime; compare instants,
    # retaining the stored ledger clock for date grouping, without changing artifacts.
    ledger_ts = ledger["ts"].cast(pl.Datetime("us")).cast(pl.Int64)
    event_ts = events["next_ts"].cast(pl.Datetime("us")).cast(pl.Int64)
    if not ledger_ts.equals(event_ts) or not ledger["ts"].is_sorted():
        raise ValueError(f"{path}: event valuation timestamps do not match sorted ledger")
    differences = {
        "event_ledger_max_diff": (events["equity_cash"] - ledger["equity_cash"]).abs().max(),
        "ledger_reconciliation_max_diff": ledger["reconciliation_diff"].abs().max(),
        "cumulative_nav_max_diff": (
            capital + ledger["net_cash_flow"].cum_sum() - ledger["equity_cash"]
        )
        .abs()
        .max(),
        "native_final_nav_diff": abs(
            float(native["final_equity_cash"]) - ledger["equity_cash"][-1]
        ),
    }
    if any(value is None or value > tolerance for value in differences.values()):
        raise ValueError(f"{path}: accounting reconciliation failed: {differences}")
    if ledger["equity_cash"].null_count() or not ledger["equity_cash"].is_finite().all():
        raise ValueError(f"{path}: non-finite ledger NAV")
    unexpected_roll_fees = 0
    if {"roll_cost_cash", "contract_roll"} <= set(events.columns):
        unexpected_roll_fees = events.filter(
            (pl.col("roll_cost_cash") != 0) & ~pl.col("contract_roll")
        ).height
        if unexpected_roll_fees:
            raise ValueError(f"{path}: roll fee charged on {unexpected_roll_fees} non-roll rows")
    audit = {
        **differences,
        "tolerance_cash": tolerance,
        "unexpected_roll_fee_rows": unexpected_roll_fees,
        "timestamp_comparison": "native serialized instants, microseconds; dates use ledger clock",
    }
    return {"config": config, "native": native, "audit": audit}, ledger, events


def ledger_returns(ledger: pl.DataFrame, capital: float) -> pl.DataFrame:
    if (ledger["equity_cash"].shift(1).fill_null(capital) <= 0).any():
        raise ValueError("returns after zero/negative NAV are undefined; inspect insolvency")
    return ledger.with_columns(
        (pl.col("equity_cash") / pl.col("equity_cash").shift(1).fill_null(capital) - 1).alias(
            "net_return"
        ),
        pl.col("net_cash_flow").alias("net_pnl"),
        pl.col("ts").alias("next_ts"),
    )


def daily_nav(ledger: pl.DataFrame, capital: float) -> pl.DataFrame:
    daily = ledger.group_by(pl.col("ts").dt.date().alias("date"), maintain_order=True).agg(
        pl.col("ts").last(),
        pl.col("equity_cash").last(),
        pl.col("net_cash_flow").sum(),
        pl.col("cost_cash").sum(),
    )
    return ledger_returns(daily, capital)


def concentration(trades: pl.DataFrame) -> dict[str, Any]:
    values = (
        [float(v) for v in trades["net_pnl"].drop_nulls()] if "net_pnl" in trades.columns else []
    )
    winners = sorted((v for v in values if v > 0), reverse=True)
    losses, net = -sum(v for v in values if v < 0), sum(values)
    return {
        "cash_profit_factor": sum(winners) / losses if losses else None,
        "attributed_trade_count": len(values),
        "top_5_share_of_net_pnl": sum(winners[:5]) / net if net > 0 else None,
        "top_10_share_of_net_pnl": sum(winners[:10]) / net if net > 0 else None,
        "net_pnl_without_best_10_diagnostic": net - sum(winners[:10]),
    }


def analyze_run(path: Path, name: str, group: str, split: date):
    audited, ledger, events = audit_run(path)
    config, native = audited["config"], audited["native"]
    capital = float(config["portfolio"]["initial_capital"])
    returns = ledger_returns(ledger, capital)
    daily = daily_nav(ledger, capital)
    trades = pl.read_parquet(path / "trades.parquet")
    row = {
        "run_id": name,
        "group": group,
        "run_path": str(path),
        "status": "completed",
        "base_currency": config["portfolio"]["base_currency"],
        "initial_capital": capital,
        "final_nav": ledger["equity_cash"][-1],
        "total_return": ledger["equity_cash"][-1] / capital - 1,
        "max_drawdown": max_drawdown([capital, *ledger["equity_cash"].to_list()]),
        "total_cost_cash": ledger["cost_cash"].sum(),
        "start_ts": str(ledger["ts"][0]),
        "end_ts": str(ledger["ts"][-1]),
        "timeframe": config["strategy"]["timeframe"],
        "filled_order_count": native["filled_order_count"],
        "partial_fill_count": native.get("partial_fill_count", 0),
        "ending_quantity": events["target_contracts"][-1],
        "active": native["filled_order_count"] > 0,
        "max_actual_leverage": (events["position_notional"].abs() / ledger["equity_cash"]).max(),
        **concentration(trades),
    }
    for key, expr in (
        ("train", pl.col("ts").dt.date() < split),
        ("check", pl.col("ts").dt.date() >= split),
    ):
        row.update(segment_metrics(key, returns.filter(expr)))
        if "contract_delta" in events.columns:
            row[key + "_fill_rows"] = events.filter(
                (
                    pl.col("next_ts").dt.date() < split
                    if key == "train"
                    else pl.col("next_ts").dt.date() >= split
                )
                & (pl.col("contract_delta") != 0)
            ).height
    annual = [
        {
            "run_id": name,
            "year": year,
            **segment_metrics("period", returns.filter(pl.col("ts").dt.year() == year)),
        }
        for year in sorted(set(ledger["ts"].dt.year()))
    ]
    return row, annual, audited["audit"], daily
