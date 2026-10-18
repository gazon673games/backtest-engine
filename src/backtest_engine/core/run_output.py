"""Atomically publish complete runs without replacing existing evidence."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from backtest_engine.core.file_io import file_transaction

RunWriter = Callable[[Path, Path], None]


def publish_run_output(
    root: Path, run_id: str, writer: RunWriter, *, reuse_identical: bool = False
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    published = root / run_id
    staging = root / f".{run_id}.staging-{uuid4().hex}"
    staging.mkdir()
    try:
        with file_transaction(root / ".publication-locks" / run_id):
            if published.exists():
                if reuse_identical:
                    writer(staging, published)
                    names = {
                        path.relative_to(staging) for path in staging.rglob("*") if path.is_file()
                    }
                    existing = {
                        path.relative_to(published)
                        for path in published.rglob("*")
                        if path.is_file()
                    }
                    if names == existing and all(
                        (staging / name).read_bytes() == (published / name).read_bytes()
                        for name in names
                    ):
                        shutil.rmtree(staging)
                        return published
                raise FileExistsError(f"Run already published: {published}; use a new run_id")
            writer(staging, published)
            staging.rename(published)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return published
