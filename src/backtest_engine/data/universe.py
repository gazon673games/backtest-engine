"""Point-in-time universe snapshots for survivorship-bias-free selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import DatasetValidationError

REQUIRED_COLUMNS = {"effective_at", "asset_id", "status"}


@dataclass(frozen=True)
class UniverseSelection:
    asset_ids: tuple[str, ...]
    records: dict[str, dict[str, Any]]


class PointInTimeUniverse:
    def __init__(
        self,
        frame: pl.DataFrame,
        *,
        ranking_column: str = "quote_volume",
        require_point_in_time: bool = True,
    ) -> None:
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise DatasetValidationError(
                "Universe dataset is missing columns: " + ", ".join(sorted(missing))
            )
        if require_point_in_time and "available_at" not in frame.columns:
            raise DatasetValidationError(
                "Universe dataset requires available_at for point-in-time selection"
            )
        if ranking_column not in frame.columns:
            raise DatasetValidationError(f"Universe ranking column is missing: {ranking_column}")
        prepared = frame.with_columns(
            pl.col("asset_id").cast(pl.String),
            pl.col("status").cast(pl.String).str.to_lowercase(),
            pl.col("effective_at").cast(pl.Datetime),
            pl.col(ranking_column).cast(pl.Float64),
        )
        if "available_at" not in prepared.columns:
            prepared = prepared.with_columns(pl.col("effective_at").alias("available_at"))
        else:
            prepared = prepared.with_columns(pl.col("available_at").cast(pl.Datetime))
        invalid_publication = prepared.filter(pl.col("available_at") < pl.col("effective_at"))
        if invalid_publication.height:
            raise DatasetValidationError("Universe available_at must not precede effective_at")
        duplicate_keys = (
            prepared.group_by(["asset_id", "effective_at", "available_at"])
            .len()
            .filter(pl.col("len") > 1)
        )
        if duplicate_keys.height:
            raise DatasetValidationError("Universe contains duplicate bitemporal records")
        self.frame = prepared.sort(["available_at", "effective_at", "asset_id"])
        self.ranking_column = ranking_column

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        ranking_column: str = "quote_volume",
        require_point_in_time: bool = True,
    ) -> PointInTimeUniverse:
        if not path.exists():
            raise DatasetValidationError(f"Universe dataset does not exist: {path}")
        if path.suffix.lower() == ".parquet":
            frame = pl.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            frame = pl.read_csv(path, try_parse_dates=True)
        else:
            raise DatasetValidationError("Universe dataset must be Parquet or CSV")
        return cls(
            frame,
            ranking_column=ranking_column,
            require_point_in_time=require_point_in_time,
        )

    def select(
        self,
        as_of: datetime,
        *,
        top_n: int | None = None,
        min_score: float | None = None,
        min_price: float | None = None,
        allowlist: set[str] | None = None,
        blocklist: set[str] | None = None,
    ) -> UniverseSelection:
        as_of = _naive_utc(as_of)
        known = self.frame.filter(
            (pl.col("available_at") <= pl.lit(as_of)) & (pl.col("effective_at") <= pl.lit(as_of))
        )
        if known.is_empty():
            return UniverseSelection((), {})
        latest = (
            known.sort(["effective_at", "available_at"])
            .group_by("asset_id", maintain_order=True)
            .last()
        )
        candidates = latest.filter(pl.col("status").is_in(["active", "listed", "trading"]))
        if min_score is not None:
            candidates = candidates.filter(pl.col(self.ranking_column) >= min_score)
        if min_price is not None:
            if "price" not in candidates.columns:
                raise DatasetValidationError("Universe min_price requires a price column")
            candidates = candidates.filter(pl.col("price") >= min_price)
        if allowlist:
            candidates = candidates.filter(pl.col("asset_id").is_in(sorted(allowlist)))
        if blocklist:
            candidates = candidates.filter(~pl.col("asset_id").is_in(sorted(blocklist)))
        candidates = candidates.sort(
            [self.ranking_column, "asset_id"],
            descending=[True, False],
            nulls_last=True,
        )
        if top_n is not None:
            candidates = candidates.head(top_n)
        records = {str(row["asset_id"]): row for row in candidates.iter_rows(named=True)}
        return UniverseSelection(tuple(records), records)


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)
