"""Losslessly compress current Workbench raw JSON and update its manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backtest_engine.adapters.raw_json import gzip_matches, write_gzip_bytes


@dataclass(frozen=True)
class RawCompactionCandidate:
    source_path: Path
    target_path: Path
    manifest_path: Path
    manifest_kind: str
    source_reference: str
    source_bytes: int
    expected_sha256: str | None


@dataclass(frozen=True)
class RawCompactionReport:
    root: Path
    applied: bool
    candidates: tuple[RawCompactionCandidate, ...]
    target_bytes: int = 0

    @property
    def source_bytes(self) -> int:
        return sum(item.source_bytes for item in self.candidates)

    @property
    def saved_bytes(self) -> int:
        return self.source_bytes - self.target_bytes if self.applied else 0


def compact_workbench_raw(data_root: Path, *, apply: bool = False) -> RawCompactionReport:
    root = data_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Workbench data root does not exist: {root}")
    candidates = tuple(plan_raw_compaction(root))
    target_bytes = 0
    if apply:
        for item in candidates:
            validate_candidate(item)
        for item in candidates:
            target_bytes += compact_candidate(item, root)
    return RawCompactionReport(root, apply, candidates, target_bytes)


def plan_raw_compaction(root: Path) -> list[RawCompactionCandidate]:
    candidates = [
        *pipeline_candidates(root),
        *file_manifest_candidates(root),
    ]
    return sorted(candidates, key=lambda item: str(item.source_path).lower())


def pipeline_candidates(root: Path) -> list[RawCompactionCandidate]:
    result = []
    for manifest_path in root.rglob("pipeline_manifest.json"):
        payload = read_manifest(manifest_path)
        reference = str(payload.get("raw_snapshot_path") or "")
        result.extend(
            make_candidate(
                root,
                manifest_path,
                manifest_path.parent.parent / "raw",
                reference,
                "pipeline",
                optional_string(payload.get("raw_snapshot_sha256")),
            )
        )
    return result


def file_manifest_candidates(root: Path) -> list[RawCompactionCandidate]:
    result = []
    for manifest_path in root.rglob("dataset_manifest.json"):
        payload = read_manifest(manifest_path)
        files = payload.get("files")
        raw = files.get("raw") if isinstance(files, dict) else None
        if not isinstance(raw, dict):
            continue
        reference = str(raw.get("path") or "")
        result.extend(
            make_candidate(
                root,
                manifest_path,
                manifest_path.parent,
                reference,
                "files",
                optional_string(raw.get("sha256")),
            )
        )
    return result


def make_candidate(
    root: Path,
    manifest_path: Path,
    raw_dir: Path,
    reference: str,
    manifest_kind: str,
    expected_sha256: str | None,
) -> list[RawCompactionCandidate]:
    name = Path(reference).name
    source = Path(reference)
    if not source.is_absolute():
        declared = manifest_path.parent / source
        source = declared if declared.is_file() else raw_dir / name
    if not name.endswith(".json") or not source.is_file():
        return []
    target = source.with_suffix(".json.gz")
    for path in (manifest_path, source, target):
        ensure_within_root(path, root)
    return [
        RawCompactionCandidate(
            source,
            target,
            manifest_path,
            manifest_kind,
            reference,
            source.stat().st_size,
            expected_sha256,
        )
    ]


def compact_candidate(item: RawCompactionCandidate, root: Path) -> int:
    contents = item.source_path.read_bytes()
    write_gzip_bytes(item.target_path, contents)
    if not gzip_matches(item.target_path, contents):
        raise ValueError(f"Compressed raw snapshot verification failed: {item.target_path}")
    update_manifest(item, root)
    item.source_path.unlink()
    return item.target_path.stat().st_size


def validate_candidate(item: RawCompactionCandidate) -> None:
    if item.expected_sha256 and file_sha256(item.source_path) != item.expected_sha256:
        raise ValueError(f"Raw snapshot checksum mismatch: {item.source_path}")


def update_manifest(item: RawCompactionCandidate, root: Path) -> None:
    ensure_within_root(item.manifest_path, root)
    payload = read_manifest(item.manifest_path)
    target_reference = f"{item.source_reference}.gz"
    target_sha256 = file_sha256(item.target_path)
    if item.manifest_kind == "pipeline":
        payload["raw_snapshot_path"] = target_reference
        payload["raw_snapshot_sha256"] = target_sha256
    else:
        payload["files"]["raw"] = {
            "path": target_reference,
            "sha256": target_sha256,
        }
    temporary = item.manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(item.manifest_path)


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_string(value: object) -> str | None:
    return str(value) if value else None


def ensure_within_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Compaction path is outside data root: {path}") from exc
