from __future__ import annotations

from typing import Any

import polars as pl


class EventFrameBuffer:
    """Convert event dictionaries to columnar chunks while the loop is running."""

    def __init__(self, schema: Any, *, chunk_size: int = 4096) -> None:
        self._schema = schema
        self._chunk_size = chunk_size
        self._pending: list[dict[str, Any]] = []
        self._chunks: list[pl.DataFrame] = []

    def append(self, row: dict[str, Any]) -> None:
        self._pending.append(row)
        if len(self._pending) >= self._chunk_size:
            self._flush()

    def finish(self) -> pl.DataFrame:
        self._flush()
        if not self._chunks:
            return pl.DataFrame(schema=self._schema)
        if len(self._chunks) == 1:
            return self._chunks[0]
        return pl.concat(self._chunks, how="vertical", rechunk=False)

    @property
    def pending_row_count(self) -> int:
        return len(self._pending)

    def _flush(self) -> None:
        if not self._pending:
            return
        self._chunks.append(pl.DataFrame(self._pending, schema=self._schema, orient="row"))
        self._pending = []
