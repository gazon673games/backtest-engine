from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backtest_engine.core.exceptions import QualityGateError
from backtest_engine.core.models import DatasetMetadata, DatasetQualityResult, QualityGateConfig

BLOCKED_STATUSES = {"split_required", "rejected", "warning", "unknown", "missing"}


@dataclass(frozen=True)
class ManifestEntry:
    raw: dict[str, Any]
    status: str
    path: str | None
    continuous_symbol: str | None
    adjustment_mode: str | None
    dataset_sha256: str | None
    reasons: list[str]


def inspect_quality_gate(
    dataset_path: Path,
    manifest_path: Path,
    metadata: DatasetMetadata,
    config: QualityGateConfig | None = None,
) -> DatasetQualityResult:
    config = config or QualityGateConfig()
    entries = _load_manifest_entries(manifest_path)
    match, matched_by = _match_manifest_entry(dataset_path, manifest_path, metadata, entries)

    if match is None:
        return DatasetQualityResult(
            dataset_path=str(dataset_path),
            manifest_path=str(manifest_path),
            status="missing",
            approved_for_backtest=False,
            matched_by=None,
            reasons=["Dataset is not present in the quality gate manifest"],
            manifest_entry=None,
        )

    binding_violations = _binding_violations(dataset_path, metadata, match)
    status = "mismatch" if binding_violations else match.status
    return DatasetQualityResult(
        dataset_path=str(dataset_path),
        manifest_path=str(manifest_path),
        status=status,
        approved_for_backtest=status == config.allowed_status,
        matched_by=matched_by,
        reasons=[*match.reasons, *binding_violations],
        manifest_entry=match.raw,
    )


def enforce_approved(quality: DatasetQualityResult) -> None:
    if quality.approved_for_backtest:
        return
    msg = f"Dataset is not approved for backtesting: status={quality.status}"
    raise QualityGateError(msg)


def _load_manifest_entries(manifest_path: Path) -> list[ManifestEntry]:
    if not manifest_path.exists():
        msg = f"Quality gate manifest does not exist: {manifest_path}"
        raise QualityGateError(msg)

    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except Exception as exc:  # pragma: no cover - json gives the concrete detail
        msg = f"Failed to read quality gate manifest: {manifest_path}"
        raise QualityGateError(msg) from exc

    raw_entries = _extract_raw_entries(manifest)
    return [_normalize_entry(entry) for entry in raw_entries]


def _extract_raw_entries(manifest: Any) -> list[dict[str, Any]]:
    if isinstance(manifest, list):
        return [entry for entry in manifest if isinstance(entry, dict)]
    if not isinstance(manifest, dict):
        return []

    for key in ("datasets", "entries", "results"):
        value = manifest.get(key)
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, dict)]

    return []


def _normalize_entry(entry: dict[str, Any]) -> ManifestEntry:
    path = _first_string(entry, "path", "dataset_path", "file_path")
    continuous_symbol = _first_string(entry, "continuous_symbol", "dataset", "symbol")
    adjustment_mode = _first_string(entry, "adjustment_mode")
    dataset_sha256 = _first_string(entry, "dataset_sha256", "sha256")

    if path is not None:
        continuous_symbol = continuous_symbol or _path_partition_value(path, "continuous_symbol")
        adjustment_mode = adjustment_mode or _path_partition_value(path, "adjustment_mode")

    status = str(entry.get("status", "unknown")).lower()
    raw_reasons = entry.get("reasons", [])
    reasons = [str(reason) for reason in raw_reasons] if isinstance(raw_reasons, list) else []

    return ManifestEntry(
        raw=entry,
        status=status,
        path=path,
        continuous_symbol=continuous_symbol,
        adjustment_mode=adjustment_mode,
        dataset_sha256=dataset_sha256,
        reasons=reasons,
    )


def _match_manifest_entry(
    dataset_path: Path,
    manifest_path: Path,
    metadata: DatasetMetadata,
    entries: list[ManifestEntry],
) -> tuple[ManifestEntry | None, str | None]:
    for mode, method in (
        ("exact", "exact_path"),
        ("absolute", "normalized_absolute_path"),
        ("relative", "normalized_relative_path"),
    ):
        path_matches = _path_matches(dataset_path, manifest_path, entries, mode=mode)
        if path_matches:
            return _single_match(path_matches, method), method

    symbol_mode_matches = [
        entry
        for entry in entries
        if entry.path is None
        if entry.continuous_symbol == metadata.continuous_symbol
        and entry.adjustment_mode == metadata.adjustment_mode
    ]
    if symbol_mode_matches:
        return _single_match(
            symbol_mode_matches,
            "continuous_symbol_and_adjustment_mode",
        ), "continuous_symbol_and_adjustment_mode"

    return None, None


def _binding_violations(
    dataset_path: Path,
    metadata: DatasetMetadata,
    entry: ManifestEntry,
) -> list[str]:
    violations: list[str] = []
    if entry.continuous_symbol and entry.continuous_symbol != metadata.continuous_symbol:
        violations.append(
            "continuous_symbol mismatch: "
            f"manifest={entry.continuous_symbol}, dataset={metadata.continuous_symbol}"
        )
    if entry.adjustment_mode and entry.adjustment_mode != metadata.adjustment_mode:
        violations.append(
            "adjustment_mode mismatch: "
            f"manifest={entry.adjustment_mode}, dataset={metadata.adjustment_mode}"
        )
    if entry.dataset_sha256 is None:
        violations.append("dataset_sha256 is required")
    elif entry.dataset_sha256 != _file_sha256(dataset_path):
        violations.append("dataset_sha256 mismatch")
    return violations


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_matches(
    dataset_path: Path,
    manifest_path: Path,
    entries: list[ManifestEntry],
    *,
    mode: str,
) -> list[ManifestEntry]:
    dataset_raw = _normalize_path_string(str(dataset_path))
    dataset_abs = _normalize_resolved_path(dataset_path)
    dataset_relatives = _dataset_relative_candidates(dataset_path, manifest_path)

    matches = []
    for entry in entries:
        if entry.path is None:
            continue
        entry_raw = _normalize_path_string(entry.path)
        if mode == "exact" and entry_raw == dataset_raw:
            matches.append(entry)
        elif mode == "absolute" and dataset_abs in _entry_absolute_candidates(
            entry.path, manifest_path
        ):
            matches.append(entry)
        elif mode == "relative" and entry_raw in dataset_relatives:
            matches.append(entry)
    return matches


def _single_match(matches: list[ManifestEntry], method: str) -> ManifestEntry:
    if len(matches) == 1:
        return matches[0]
    symbols = sorted({str(match.continuous_symbol) for match in matches})
    modes = sorted({str(match.adjustment_mode) for match in matches})
    msg = (
        "Ambiguous quality gate manifest entries for dataset "
        f"using {method}: symbols={symbols}, adjustment_modes={modes}"
    )
    raise QualityGateError(msg)


def _entry_absolute_candidates(entry_path: str, manifest_path: Path) -> set[str]:
    raw_path = Path(entry_path)
    candidates = [raw_path]

    if not raw_path.is_absolute():
        candidates.append(Path.cwd() / raw_path)
        candidates.extend(parent / raw_path for parent in _manifest_candidate_roots(manifest_path))

    return {_normalize_resolved_path(candidate) for candidate in candidates}


def _dataset_relative_candidates(dataset_path: Path, manifest_path: Path) -> set[str]:
    dataset_abs = dataset_path.resolve(strict=False)
    candidates = {_normalize_path_string(str(dataset_path))}
    for root in [Path.cwd(), *_manifest_candidate_roots(manifest_path)]:
        try:
            relative = dataset_abs.relative_to(root.resolve(strict=False))
        except ValueError:
            continue
        candidates.add(_normalize_path_string(str(relative)))
    return candidates


def _manifest_candidate_roots(manifest_path: Path) -> list[Path]:
    manifest_abs = manifest_path.resolve(strict=False)
    return list(manifest_abs.parents)


def _normalize_resolved_path(path: Path) -> str:
    return _normalize_path_string(str(path.resolve(strict=False)))


def _normalize_path_string(path: str) -> str:
    normalized = path.replace("\\", "/").strip().rstrip("/")
    normalized = re.sub(r"/+", "/", normalized)
    if os.name == "nt":
        normalized = normalized.lower()
    return normalized


def _path_partition_value(path: str, key: str) -> str | None:
    for part in re.split(r"[\\/]+", path):
        if part.startswith(f"{key}="):
            return part.split("=", 1)[1]
    return None


def _first_string(entry: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = entry.get(key)
        if value is not None:
            return str(value)
    return None
