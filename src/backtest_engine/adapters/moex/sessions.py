from __future__ import annotations

from zoneinfo import ZoneInfo

MOEX_TIMEZONE = ZoneInfo("Europe/Moscow")

MOEX_SESSION_WINDOWS = (
    ("main_am", "06:50", "14:00"),
    ("main_pm", "14:05", "18:50"),
    ("evening", "19:05", "23:50"),
)


def session_window_payloads() -> list[dict[str, str]]:
    return [{"id": name, "start": start, "end": end} for name, start, end in MOEX_SESSION_WINDOWS]
