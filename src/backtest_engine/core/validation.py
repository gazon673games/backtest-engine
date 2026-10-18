from __future__ import annotations


def validate_choice(value: str, choices: frozenset[str], label: str) -> str:
    normalized = value.strip().lower()
    if normalized not in choices:
        msg = f"Invalid {label}: {value}. Expected one of: {', '.join(sorted(choices))}"
        raise ValueError(msg)
    return normalized
