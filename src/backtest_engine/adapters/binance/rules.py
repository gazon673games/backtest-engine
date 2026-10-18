"""Point-in-time Binance filters, fees, and perpetual margin schedules."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.instrument_registry import MarginTierSpec

RULE_ENFORCEMENTS = frozenset({"off", "audit", "strict"})


class BinanceRuleSnapshot(BaseModel):
    valid_from: date
    valid_to: date | None = None
    source_as_of: datetime | None = None
    tick_size: float | None = Field(default=None, gt=0.0)
    quantity_step: float | None = Field(default=None, gt=0.0)
    minimum_quantity: float | None = Field(default=None, ge=0.0)
    minimum_notional: float | None = Field(default=None, ge=0.0)
    maker_fee_bps: float | None = Field(default=None, ge=0.0)
    taker_fee_bps: float | None = Field(default=None, ge=0.0)
    liquidation_fee_bps: float | None = Field(default=None, ge=0.0)
    margin_tiers: list[MarginTierSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_dates(self) -> BinanceRuleSnapshot:
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("Binance rule valid_to must not precede valid_from")
        return self


class BinanceRuleArchive(BaseModel):
    schema_version: int = 1
    venue: str = "binance"
    product: str
    symbol: str
    snapshots: list[BinanceRuleSnapshot]

    @model_validator(mode="after")
    def _validate_snapshots(self) -> BinanceRuleArchive:
        if not self.snapshots:
            raise ValueError("Binance rule archive must contain snapshots")
        ordered = sorted(self.snapshots, key=lambda item: item.valid_from)
        if ordered != self.snapshots:
            raise ValueError("Binance rule snapshots must be ordered by valid_from")
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.valid_to is None or current.valid_from <= previous.valid_to:
                raise ValueError("Binance rule snapshots overlap or have an open-ended predecessor")
        return self


def load_rule_archive(path: Path, *, product: str, symbol: str) -> BinanceRuleArchive:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        archive = BinanceRuleArchive(**payload)
    except Exception as exc:
        raise DatasetValidationError(f"Invalid Binance rule archive: {path}: {exc}") from exc
    if archive.venue.lower() != "binance":
        raise DatasetValidationError(f"Rule archive venue must be binance: {archive.venue}")
    if archive.product != product or archive.symbol.upper() != symbol.upper():
        raise DatasetValidationError(
            "Rule archive identity mismatch: "
            f"expected {product}/{symbol}, got {archive.product}/{archive.symbol}"
        )
    return archive


def build_rule_registry(
    *,
    product: str,
    symbol: str,
    metadata_spec: dict[str, Any],
    rules_path: Path | None,
    enforcement: str,
    from_date: str,
    till_date: str,
    retrieved_at: datetime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = enforcement.strip().lower()
    if normalized not in RULE_ENFORCEMENTS:
        raise DatasetValidationError(
            f"rules_enforcement must be one of: {', '.join(sorted(RULE_ENFORCEMENTS))}"
        )
    if rules_path is None:
        violations = missing_archive_violations(product)
        specs = [
            {
                **metadata_spec,
                "venue": "binance",
                "rule_source": "current_exchange_info",
                "source_as_of": retrieved_at.isoformat(),
            }
        ]
    else:
        archive = load_rule_archive(rules_path, product=product, symbol=symbol)
        violations = archive_violations(
            archive,
            product=product,
            from_date=date.fromisoformat(from_date),
            till_date=date.fromisoformat(till_date),
        )
        specs = [snapshot_spec(metadata_spec, item, rules_path) for item in archive.snapshots]
    status = "not_required" if normalized == "off" else ("warning" if violations else "approved")
    audit = {
        "status": status,
        "enforcement": normalized,
        "rules_path": str(rules_path) if rules_path else None,
        "snapshot_count": len(specs),
        "violations": violations,
    }
    if normalized == "strict" and violations:
        raise DatasetValidationError("Binance point-in-time rules failed: " + "; ".join(violations))
    return specs, audit


def snapshot_spec(
    metadata_spec: dict[str, Any],
    snapshot: BinanceRuleSnapshot,
    rules_path: Path,
) -> dict[str, Any]:
    overrides = snapshot.model_dump(mode="json", exclude_none=True)
    overrides.update(
        {
            "venue": "binance",
            "rule_source": str(rules_path),
            "source_as_of": (
                snapshot.source_as_of
                or datetime.combine(snapshot.valid_from, datetime.min.time(), UTC)
            ).isoformat(),
        }
    )
    return {**metadata_spec, **overrides}


def archive_violations(
    archive: BinanceRuleArchive,
    *,
    product: str,
    from_date: date,
    till_date: date,
) -> list[str]:
    violations: list[str] = []
    cursor = from_date
    for snapshot in archive.snapshots:
        if snapshot.valid_to is not None and snapshot.valid_to < from_date:
            continue
        if snapshot.valid_from > till_date:
            break
        if snapshot.valid_from > cursor:
            gap_end = snapshot.valid_from - timedelta(days=1)
            violations.append(f"rules gap {cursor.isoformat()}..{gap_end.isoformat()}")
        end = min(snapshot.valid_to or till_date, till_date)
        if snapshot.valid_from <= cursor <= end:
            cursor = end + timedelta(days=1)
        violations.extend(snapshot_violations(snapshot, product))
    if cursor <= till_date:
        violations.append(f"rules do not cover till_date={till_date.isoformat()}")
    return sorted(set(violations))


def snapshot_violations(snapshot: BinanceRuleSnapshot, product: str) -> list[str]:
    prefix = snapshot.valid_from.isoformat()
    missing = [
        name
        for name in (
            "tick_size",
            "quantity_step",
            "minimum_quantity",
            "minimum_notional",
            "maker_fee_bps",
            "taker_fee_bps",
        )
        if getattr(snapshot, name) is None
    ]
    if product == "perpetual" and not snapshot.margin_tiers:
        missing.append("margin_tiers")
    if product == "perpetual":
        if snapshot.liquidation_fee_bps is None:
            missing.append("liquidation_fee_bps")
        if any(tier.max_leverage is None for tier in snapshot.margin_tiers):
            missing.append("margin_tiers.max_leverage")
    return [f"snapshot {prefix} missing {name}" for name in missing]


def missing_archive_violations(product: str) -> list[str]:
    missing = ["historical fee schedule", "historical exchange filters"]
    if product == "perpetual":
        missing.append("historical leverage/margin brackets")
    return ["point-in-time rule archive missing", *missing]
