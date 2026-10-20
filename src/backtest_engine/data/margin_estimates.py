"""Explicit, causal research estimates; these are not exchange clearing margins."""

from __future__ import annotations

from bisect import bisect_right
from datetime import date, datetime
from decimal import ROUND_CEILING, Decimal, localcontext
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backtest_engine.core.exceptions import DatasetValidationError


class MarginEstimateInterval(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    valid_from: datetime
    valid_to: datetime
    available_at: datetime
    rate: float = Field(gt=0, le=1)
    maximum_contracts: float = Field(gt=0)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_trade_date: date | None = None
    carried_forward: bool = False

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        timestamps = (self.valid_from, self.valid_to, self.available_at)
        if any(value.tzinfo is not None for value in timestamps):
            raise ValueError("Margin estimate times must use the dataset's naive local clock")
        if self.valid_to <= self.valid_from or self.available_at > self.valid_from:
            raise ValueError("Margin interval must be nonempty and available before it starts")
        return self


def resolve_margin_estimate(spec: Any, row: dict[str, Any], current_ts: datetime) -> Any:
    schedule = spec.margin_estimate_schedule
    if not schedule:
        return spec
    if current_ts.tzinfo is not None:
        raise DatasetValidationError("Margin estimate schedule requires naive dataset timestamps")
    index = bisect_right(schedule, current_ts, key=lambda item: item.valid_from) - 1
    if index < 0 or current_ts >= schedule[index].valid_to:
        raise DatasetValidationError(
            f"Historical margin estimate is unavailable for {spec.symbol} at {current_ts}; "
            "rebuild the dataset with risk history covering this time or supply exact rules"
        )
    snapshot = schedule[index]
    price = Decimal(str(row["close"]))
    if not price.is_finite() or price <= 0:
        raise DatasetValidationError(f"Margin estimate requires a positive price for {spec.symbol}")
    # Research convention: current contract notional * historical first-level risk rate.
    # It excludes clearing price scenarios, interest risk and broker/client add-ons.
    with localcontext(prec=34):
        point_value = Decimal(str(spec.contract_multiplier)) * Decimal(str(spec.lot_size))
        if spec.tick_size is not None and spec.tick_value is not None:
            point_value = Decimal(str(spec.tick_value)) / Decimal(str(spec.tick_size))
        margin = (price * point_value * Decimal(str(snapshot.rate))).quantize(
            Decimal("0.01"), rounding=ROUND_CEILING
        )
    return spec.model_copy(
        update={"initial_margin": float(margin), "maximum_quantity": snapshot.maximum_contracts}
    )
