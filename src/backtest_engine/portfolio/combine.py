from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from backtest_engine.core.exceptions import BacktestError


class PortfolioCombineConfig(BaseModel):
    run_paths: list[Path]
    output_path: Path
    initial_capital: float | None = Field(default=None, gt=0.0)


class PortfolioCombineResult(BaseModel):
    output_path: Path
    run_count: int
    event_count: int
    final_equity_cash: float
    total_return: float
    metrics_path: Path


def combine_intraday_event_portfolio(config: PortfolioCombineConfig) -> PortfolioCombineResult:
    if not config.run_paths:
        raise BacktestError("run_paths must not be empty")

    currencies = {_read_base_currency(path) for path in config.run_paths}
    if len(currencies) > 1:
        raise BacktestError(f"Portfolio combine requires one base currency: {sorted(currencies)}")
    run_frames = []
    inferred_capital = 0.0
    base_currency = None
    for run_path in config.run_paths:
        currency = _read_base_currency(run_path)
        if base_currency is not None and currency != base_currency:
            raise BacktestError(
                f"Portfolio combine requires one base currency: {base_currency} vs {currency} "
                f"in {run_path}; use the multi runtime with explicit FX"
            )
        base_currency = currency
        events_path = run_path / "events.parquet"
        if not events_path.exists():
            raise BacktestError(f"Missing events.parquet in run path: {run_path}")
        events = pl.read_parquet(events_path)
        if events.height == 0:
            continue
        run_id = run_path.name
        continuous_symbol = _read_continuous_symbol(run_path, events)
        inferred_capital += _read_initial_capital(run_path)
        run_frames.append(
            events.select(
                "ts",
                "next_ts",
                "net_pnl",
                "gross_pnl",
                "cost_cash",
                "commission_cost_cash",
                "slippage_cost_cash",
                "roll_cost_cash",
                "position_notional",
                "turnover_notional",
            ).with_columns(
                pl.lit(run_id).alias("run_id"),
                pl.lit(continuous_symbol).alias("continuous_symbol"),
            )
        )

    initial_capital = config.initial_capital or inferred_capital
    if initial_capital <= 0.0:
        raise BacktestError("initial_capital must be positive or inferable from run configs")
    if not run_frames:
        legs = _empty_legs_frame()
        portfolio = _empty_portfolio_frame()
    else:
        legs = pl.concat(run_frames, how="diagonal_relaxed")
        portfolio = _portfolio_equity_from_legs(legs, initial_capital)

    output_dir = config.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    legs_path = output_dir / "portfolio_legs.parquet"
    equity_path = output_dir / "portfolio_equity.parquet"
    legs.write_parquet(legs_path)
    legs.write_csv(legs_path.with_suffix(".csv"))
    portfolio.write_parquet(equity_path)
    portfolio.write_csv(equity_path.with_suffix(".csv"))

    metrics = _portfolio_metrics(portfolio, initial_capital)
    metrics["base_currency"] = base_currency
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    _write_json(
        output_dir / "portfolio_manifest.json",
        {
            "manifest_version": 1,
            "run_paths": [str(path) for path in config.run_paths],
            "run_hashes": {
                str(path): _file_sha256(path / "experiment_manifest.json")
                for path in config.run_paths
                if (path / "experiment_manifest.json").exists()
            },
            "initial_capital": initial_capital,
            "base_currency": base_currency,
            "files": {
                "portfolio_legs": str(legs_path),
                "portfolio_equity": str(equity_path),
                "metrics": str(metrics_path),
            },
            "metrics": metrics,
        },
    )
    return PortfolioCombineResult(
        output_path=output_dir,
        run_count=len(config.run_paths),
        event_count=portfolio.height,
        final_equity_cash=float(metrics["final_equity_cash"]),
        total_return=float(metrics["total_return"]),
        metrics_path=metrics_path,
    )


def _portfolio_equity_from_legs(legs: pl.DataFrame, initial_capital: float) -> pl.DataFrame:
    grouped_rows: list[dict[str, Any]] = []
    for ts, frame in legs.group_by("ts", maintain_order=True):
        ts_value = ts[0] if isinstance(ts, tuple) else ts
        grouped_rows.append(
            {
                "ts": ts_value,
                "net_pnl": float(frame.get_column("net_pnl").sum()),
                "gross_pnl": float(frame.get_column("gross_pnl").sum()),
                "cost_cash": float(frame.get_column("cost_cash").sum()),
                "commission_cost_cash": float(frame.get_column("commission_cost_cash").sum()),
                "slippage_cost_cash": float(frame.get_column("slippage_cost_cash").sum()),
                "roll_cost_cash": float(frame.get_column("roll_cost_cash").sum()),
                "gross_position_notional": float(frame.get_column("position_notional").sum()),
                "turnover_notional": float(frame.get_column("turnover_notional").sum()),
                "active_leg_count": int(
                    sum(1 for value in frame.get_column("position_notional").to_list() if value)
                ),
            }
        )
    grouped_rows.sort(key=lambda row: row["ts"])
    equity_cash = initial_capital
    for row in grouped_rows:
        equity_cash += float(row["net_pnl"])
        row["equity_cash"] = equity_cash
        row["equity"] = equity_cash / initial_capital
        row["net_return"] = float(row["net_pnl"]) / (equity_cash - float(row["net_pnl"]))
    return pl.DataFrame(grouped_rows, orient="row")


def _read_base_currency(run_path: Path) -> str:
    path = run_path / "config.json"
    if not path.is_file():
        raise BacktestError(
            f"Portfolio combine requires config.json with base_currency: {run_path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    portfolio = payload.get("portfolio", payload.get("config", {}).get("portfolio", {}))
    currency = portfolio.get("base_currency")
    if not currency:
        raise BacktestError(f"Portfolio combine requires an explicit base_currency: {run_path}")
    return str(currency).upper()


def _portfolio_metrics(portfolio: pl.DataFrame, initial_capital: float) -> dict[str, Any]:
    if portfolio.height == 0:
        return {
            "event_count": 0,
            "initial_capital": initial_capital,
            "final_equity_cash": initial_capital,
            "total_return": 0.0,
            "net_pnl": 0.0,
            "total_cost_cash": 0.0,
            "max_gross_position_notional": 0.0,
            "max_drawdown": 0.0,
        }
    equity_values = [float(value) for value in portfolio.get_column("equity").to_list()]
    equity_values.insert(0, 1.0)
    return {
        "event_count": portfolio.height,
        "initial_capital": initial_capital,
        "start_ts": _to_iso(portfolio.get_column("ts")[0]),
        "end_ts": _to_iso(portfolio.get_column("ts")[-1]),
        "final_equity_cash": float(portfolio.get_column("equity_cash")[-1]),
        "total_return": equity_values[-1] - 1.0,
        "net_pnl": float(portfolio.get_column("net_pnl").sum()),
        "gross_pnl": float(portfolio.get_column("gross_pnl").sum()),
        "total_cost_cash": float(portfolio.get_column("cost_cash").sum()),
        "max_gross_position_notional": float(portfolio.get_column("gross_position_notional").max()),
        "turnover_notional": float(portfolio.get_column("turnover_notional").sum()),
        "max_drawdown": _max_drawdown(equity_values),
    }


def _read_initial_capital(run_path: Path) -> float:
    config_path = run_path / "config.json"
    if not config_path.exists():
        return 0.0
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if "portfolio" in payload:
        return float(payload.get("portfolio", {}).get("initial_capital", 0.0))
    return float(payload.get("config", {}).get("portfolio", {}).get("initial_capital", 0.0))


def _read_continuous_symbol(run_path: Path, events: pl.DataFrame) -> str:
    validation_path = run_path / "dataset_validation.json"
    if validation_path.exists():
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        if payload.get("continuous_symbol"):
            return str(payload["continuous_symbol"])
    if "active_contract" in events.columns and events.height > 0:
        return str(events.get_column("active_contract")[0])
    return run_path.name


def _empty_legs_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [],
        schema={
            "ts": pl.Datetime,
            "next_ts": pl.Datetime,
            "net_pnl": pl.Float64,
            "gross_pnl": pl.Float64,
            "cost_cash": pl.Float64,
            "commission_cost_cash": pl.Float64,
            "slippage_cost_cash": pl.Float64,
            "roll_cost_cash": pl.Float64,
            "position_notional": pl.Float64,
            "turnover_notional": pl.Float64,
            "run_id": pl.String,
            "continuous_symbol": pl.String,
        },
    )


def _empty_portfolio_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [],
        schema={
            "ts": pl.Datetime,
            "net_pnl": pl.Float64,
            "gross_pnl": pl.Float64,
            "cost_cash": pl.Float64,
            "commission_cost_cash": pl.Float64,
            "slippage_cost_cash": pl.Float64,
            "roll_cost_cash": pl.Float64,
            "gross_position_notional": pl.Float64,
            "turnover_notional": pl.Float64,
            "active_leg_count": pl.Int64,
            "equity_cash": pl.Float64,
            "equity": pl.Float64,
            "net_return": pl.Float64,
        },
    )


def _max_drawdown(equity_values: list[float]) -> float:
    peak = equity_values[0]
    max_drawdown = 0.0
    for value in equity_values:
        peak = max(peak, value)
        drawdown = value / peak - 1.0 if peak != 0.0 else 0.0
        max_drawdown = min(max_drawdown, drawdown)
    return max_drawdown


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _to_iso(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
