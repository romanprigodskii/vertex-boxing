"""Normalize stance strings to the stance enum."""

from __future__ import annotations


def normalize_stance(raw: str | None) -> str:
    if not raw:
        return "unknown"
    s = raw.strip().lower()
    if "south" in s:
        return "southpaw"
    if "ortho" in s:
        return "orthodox"
    if "switch" in s or "convert" in s:
        return "switch"
    return "unknown"
