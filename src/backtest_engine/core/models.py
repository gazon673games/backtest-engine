from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

StrategyName = Literal["buy_and_hold", "moving_average_crossover", "plugin"]


class DatasetConfig(BaseModel):
    path: Path
    quality_manifest: Path


class QualityGateConfig(BaseModel):
    allowed_status: str = "approved"


class StrategyConfig(BaseModel):
    name: StrategyName
    strategy_type: Literal["builtin", "plugin"] = "builtin"
    fast_window: int | None = None
    slow_window: int | None = None
    strategy_file: Path | None = None
    strategy_file_sha256: str | None = None
    strategy_params: dict[str, str] = Field(default_factory=dict)
    lookahead_tolerance: float = Field(default=1e-12, ge=0.0)

    @model_validator(mode="after")
    def validate_strategy_parameters(self) -> StrategyConfig:
        if self.name == "plugin":
            self.strategy_type = "plugin"
            if self.strategy_file is None:
                msg = "plugin strategy requires strategy_file"
                raise ValueError(msg)
            return self

        self.strategy_type = "builtin"
        if self.name != "moving_average_crossover":
            return self

        if self.fast_window is None or self.slow_window is None:
            msg = "moving_average_crossover requires fast_window and slow_window"
            raise ValueError(msg)
        if self.fast_window <= 0:
            msg = "fast_window must be greater than 0"
            raise ValueError(msg)
        if self.slow_window <= 0:
            msg = "slow_window must be greater than 0"
            raise ValueError(msg)
        if self.fast_window >= self.slow_window:
            msg = "fast_window must be less than slow_window"
            raise ValueError(msg)
        return self


class ExecutionConfig(BaseModel):
    timing: str = "signal_at_close_apply_next_close"
    commission_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)
    trading_days_per_year: int | None = Field(default=None, gt=0)


class OutputConfig(BaseModel):
    root: Path = Path("data/backtests")
    run_id: str = Field(min_length=1)
    write_csv: bool = False


class BacktestConfig(BaseModel):
    dataset: DatasetConfig
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    strategy: StrategyConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig


class DatasetMetadata(BaseModel):
    path: str
    continuous_symbol: str
    adjustment_mode: str
    row_count: int
    min_ts: str
    max_ts: str


class DatasetQualityResult(BaseModel):
    dataset_path: str
    manifest_path: str
    status: str
    approved_for_backtest: bool
    matched_by: str | None = None
    reasons: list[str] = Field(default_factory=list)
    manifest_entry: dict[str, Any] | None = None


class DatasetInspection(BaseModel):
    metadata: DatasetMetadata
    quality: DatasetQualityResult


class PerformanceMetrics(BaseModel):
    start_ts: str
    end_ts: str
    bar_count: int
    total_return: float
    annualized_return: float | None
    annualized_volatility: float | None
    sharpe: float | None
    max_drawdown: float
    trade_count: int
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    profit_factor: float | None
    best_trade: float | None
    worst_trade: float | None
    avg_trade_return: float | None
    avg_holding_bars: float | None
    long_trade_count: int
    closed_by_end_of_data_count: int
    exposure: float
    mean_daily_return: float
    daily_volatility: float
    best_day: float
    worst_day: float


class BacktestResult(BaseModel):
    output_dir: Path
    metrics: PerformanceMetrics
    dataset_quality: DatasetQualityResult
