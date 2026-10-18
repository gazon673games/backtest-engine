"""Asset-neutral OHLCV profiles and explicit compatibility aliases for event runners."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError

COMMON_MARKET_COLUMNS = frozenset(
    {
        "source",
        "exchange",
        "engine",
        "market",
        "board",
        "symbol",
        "timeframe",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
        "currency",
        "dataset_type",
    }
)


@dataclass(frozen=True)
class DatasetSchemaProfile:
    dataset_type: str
    asset_class: str
    instrument_type: str
    required_columns: frozenset[str] = frozenset()


SCHEMA_PROFILES = {
    "nearest_contract": DatasetSchemaProfile(
        "nearest_contract",
        "futures",
        "future",
        frozenset(
            {
                "continuous_symbol",
                "active_contract",
                "source_contract_symbol",
                "adjustment_mode",
            }
        ),
    ),
    "spot": DatasetSchemaProfile(
        "spot",
        "generic",
        "spot",
        frozenset({"base_currency", "quote_currency"}),
    ),
    "perpetual": DatasetSchemaProfile(
        "perpetual",
        "generic",
        "perpetual",
        frozenset({"base_currency", "quote_currency", "settlement_currency"}),
    ),
    "ohlcv": DatasetSchemaProfile("ohlcv", "generic", "spot"),
}


def normalize_intraday_asset_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Add documented event-engine aliases without changing prices or timestamps."""
    dataset_type = single_text(frame, "dataset_type")
    profile = schema_profile(dataset_type)
    additions: list[pl.Expr] = []
    aliases = {
        "asset_class": pl.lit(profile.asset_class),
        "instrument_type": pl.lit(profile.instrument_type),
    }
    if profile.dataset_type != "nearest_contract":
        aliases.update(
            {
                "continuous_symbol": pl.col("symbol"),
                "active_contract": pl.col("symbol"),
                "source_contract_symbol": pl.col("symbol"),
                "adjustment_mode": pl.lit("unadjusted"),
            }
        )
    for column, expression in aliases.items():
        if column not in frame.columns:
            additions.append(expression.alias(column))
    return frame.with_columns(additions) if additions else frame


def schema_profile(dataset_type: str) -> DatasetSchemaProfile:
    try:
        return SCHEMA_PROFILES[dataset_type]
    except KeyError as exc:
        supported = ", ".join(sorted(SCHEMA_PROFILES))
        raise DatasetValidationError(
            f"Unsupported dataset_type={dataset_type!r}; expected one of: {supported}"
        ) from exc


def validate_profile_columns(frame: pl.DataFrame, profile: DatasetSchemaProfile) -> None:
    required = COMMON_MARKET_COLUMNS | profile.required_columns
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DatasetValidationError(
            f"{profile.dataset_type} dataset is missing required columns: {', '.join(missing)}"
        )


def compatibility_columns(frame: pl.DataFrame) -> list[str]:
    return [
        column
        for column in (
            "continuous_symbol",
            "active_contract",
            "source_contract_symbol",
            "adjustment_mode",
            "asset_class",
            "instrument_type",
        )
        if column in frame.columns
    ]


def single_text(frame: pl.DataFrame, column: str) -> str:
    if column not in frame.columns:
        raise DatasetValidationError(f"Dataset is missing required column: {column}")
    values = frame.get_column(column).drop_nulls().unique().to_list()
    if len(values) != 1:
        raise DatasetValidationError(f"{column} must contain exactly one non-null value")
    return str(values[0]).strip().lower()
