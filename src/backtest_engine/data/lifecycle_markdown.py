"""Render compact Markdown tables for data lifecycle audit artifacts."""

from __future__ import annotations

from typing import Any

import polars as pl


def markdown_table(frame: pl.DataFrame) -> str:
    if frame.height == 0 or not frame.columns:
        return "_No rows._"
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(format_cell(value) for value in row) + " |" for row in frame.iter_rows()
    ]
    return "\n".join([header, divider, *rows])


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")
