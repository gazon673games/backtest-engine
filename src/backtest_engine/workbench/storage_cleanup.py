"""Plans and applies conservative cleanup of duplicated Workbench artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CleanupCandidate:
    path: Path
    category: str
    size_bytes: int


@dataclass(frozen=True)
class CleanupReport:
    root: Path
    applied: bool
    candidates: tuple[CleanupCandidate, ...]

    @property
    def total_bytes(self) -> int:
        return sum(candidate.size_bytes for candidate in self.candidates)

    def category_totals(self) -> dict[str, tuple[int, int]]:
        totals: dict[str, tuple[int, int]] = {}
        for candidate in self.candidates:
            count, size = totals.get(candidate.category, (0, 0))
            totals[candidate.category] = count + 1, size + candidate.size_bytes
        return totals


def cleanup_workbench_storage(data_root: Path, *, apply: bool = False) -> CleanupReport:
    root = data_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Workbench data root does not exist: {root}")
    candidates = tuple(plan_cleanup(root))
    if apply:
        for candidate in candidates:
            ensure_within_root(candidate.path, root)
            candidate.path.unlink(missing_ok=True)
            remove_empty_legacy_directory(candidate)
    return CleanupReport(root=root, applied=apply, candidates=candidates)


def plan_cleanup(root: Path) -> list[CleanupCandidate]:
    candidates = [*paired_csv_candidates(root), *raw_snapshot_candidates(root)]
    by_path = {candidate.path.resolve(): candidate for candidate in candidates}
    return sorted(by_path.values(), key=lambda candidate: str(candidate.path).lower())


def paired_csv_candidates(root: Path) -> list[CleanupCandidate]:
    result = []
    for csv_path in root.rglob("*.csv"):
        parquet_path = csv_path.with_suffix(".parquet")
        if parquet_path.is_file():
            result.append(candidate(csv_path, "paired_csv"))
    return result


def raw_snapshot_candidates(root: Path) -> list[CleanupCandidate]:
    result = []
    for manifest_path in root.rglob("pipeline_manifest.json"):
        payload = read_json(manifest_path)
        current_name = Path(str(payload.get("raw_snapshot_path") or "")).name
        dataset_root = manifest_path.parent.parent
        raw_dir = dataset_root / "raw"
        current_path = raw_dir / current_name
        if not current_name or not current_path.is_file():
            continue
        result.extend(
            candidate(path, "superseded_raw")
            for path in raw_dir.glob("*.json")
            if path.name != current_name
        )
        legacy = dataset_root / "adapter_raw" / "candles_raw.json"
        if legacy.is_file():
            result.append(candidate(legacy, "legacy_adapter_raw"))
    return result


def candidate(path: Path, category: str) -> CleanupCandidate:
    return CleanupCandidate(path=path, category=category, size_bytes=path.stat().st_size)


def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def ensure_within_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Cleanup candidate is outside data root: {path}") from exc


def remove_empty_legacy_directory(candidate: CleanupCandidate) -> None:
    if candidate.category != "legacy_adapter_raw":
        return
    try:
        candidate.path.parent.rmdir()
    except OSError:
        pass
