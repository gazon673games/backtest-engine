from __future__ import annotations

import json
from datetime import date, datetime, time
from typing import Any


def parse_features(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return raw_value
    if raw_value in (None, ""):
        return {}
    try:
        payload = json.loads(str(raw_value))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def json_value(value: Any) -> Any:
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    return value
