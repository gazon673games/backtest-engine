"""Content identity for daily batch cases."""

from pathlib import Path
from typing import Any

from backtest_engine.core.run_identity import case_identity
from backtest_engine.daily.batch_io import safe_path_part


def generate_run_id(
    *,
    continuous_symbol: str,
    adjustment_mode: str,
    strategy_file: Path,
    params: dict[str, int],
    dataset_path: Path,
    execution_params: dict[str, Any],
) -> str:
    strategy_name = safe_path_part(strategy_file.stem)
    param_part = "_".join(
        f"{safe_path_part(key)}-{safe_path_part(str(value))}"
        for key, value in sorted(params.items())
    )
    base = "_".join(
        part
        for part in (
            safe_path_part(continuous_symbol),
            safe_path_part(adjustment_mode),
            strategy_name,
            param_part,
        )
        if part
    )
    digest = case_identity(
        dataset_path,
        strategy_file,
        {
            "params": params,
            "execution": execution_params,
            "symbol": continuous_symbol,
            "adjustment": adjustment_mode,
        },
    )
    return f"{base[:163].rstrip('_')}_{digest}"
