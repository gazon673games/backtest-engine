from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import Any, overload

import polars as pl


class FrameRows(Sequence[dict[str, Any]]):
    """Expose Polars rows on demand without duplicating the whole frame in Python."""

    def __init__(self, frame: pl.DataFrame, *, cache_size: int = 32) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self._frame = frame
        self._cache_size = cache_size
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return self._frame.height

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        cached = self._cache.pop(index, None)
        if cached is not None:
            self._cache[index] = cached
            return cached
        row = self._frame.row(index, named=True)
        self._cache[index] = row
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return row

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    @property
    def cached_row_count(self) -> int:
        return len(self._cache)
