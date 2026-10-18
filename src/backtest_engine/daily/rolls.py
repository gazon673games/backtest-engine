from __future__ import annotations

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.models import DatasetMetadata


def validate_roll_returns(
    metadata: DatasetMetadata,
    active_contracts: list[str],
    positions: list[float],
) -> None:
    has_exposed_roll = any(
        previous != current and position != 0.0
        for previous, current, position in zip(
            active_contracts[:-1],
            active_contracts[1:],
            positions[1:],
            strict=True,
        )
    )
    if has_exposed_roll and metadata.adjustment_mode == "unadjusted":
        raise DatasetValidationError(
            "Daily PnL across a futures roll requires an adjusted continuous series; "
            "unadjusted contract gaps are not tradable returns"
        )


def roll_turnover(position: list[float], active_contracts: list[str]) -> list[float]:
    turnover = [0.0]
    for index in range(1, len(position)):
        changed = active_contracts[index - 1] != active_contracts[index]
        turnover.append(2.0 * abs(position[index]) if changed else 0.0)
    return turnover
