"""Best-effort runtime provenance embedded into reproducible manifests."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("backtest-engine", "polars", "pyarrow", "pydantic", "tzdata")


def runtime_provenance() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": package_versions(),
        "source": source_identity(repository),
        "dependency_lock_sha256": optional_file_sha256(repository / "uv.lock"),
        "byteorder": sys.byteorder,
    }


def package_versions() -> dict[str, str | None]:
    result = {}
    for package in TRACKED_PACKAGES:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def source_identity(repository: Path) -> dict[str, Any]:
    commit = git_output(repository, "rev-parse", "HEAD")
    status = git_output(repository, "status", "--porcelain", "--untracked-files=normal")
    return {
        "repository": str(repository),
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
    }


def git_output(repository: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def optional_file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
