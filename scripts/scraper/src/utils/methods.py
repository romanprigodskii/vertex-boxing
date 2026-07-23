"""Normalize boxing result-method strings to the bout_method enum.

Also classifies a method into 'ko' (any stoppage), 'decision', 'draw', or
'nc' — the coarse buckets the model's finish-for/finish-against and durability
features actually use.
"""

from __future__ import annotations

# (substring to look for, enum value). Order matters — check the specific
# decision splits before the bare "decision"/"pts", and "technical" before
# plain draw/decision.
_RULES: tuple[tuple[str, str], ...] = (
    ("technical decision", "technical_decision"),
    ("technical draw", "technical_draw"),
    ("unanimous", "ud"),
    ("majority draw", "draw"),
    ("split draw", "draw"),
    ("majority", "md"),
    ("split", "sd"),
    ("no contest", "nc"),
    ("retirement", "rtd"),
    ("corner", "rtd"),
    ("disqualif", "dq"),
    ("tko", "tko"),
    ("rtd", "rtd"),
    ("ko", "ko"),
    ("draw", "draw"),
    ("decision", "pts"),
    ("points", "pts"),
    ("pts", "pts"),
    ("dq", "dq"),
    ("nc", "nc"),
)

# enum value → coarse bucket used by features.
_BUCKET: dict[str, str] = {
    "ko": "ko",
    "tko": "ko",
    "rtd": "ko",
    "ud": "decision",
    "sd": "decision",
    "md": "decision",
    "pts": "decision",
    "technical_decision": "decision",
    "draw": "draw",
    "technical_draw": "draw",
    "dq": "dq",
    "nc": "nc",
}


def normalize_method(raw: str | None) -> str | None:
    """Return a bout_method enum value, or None if unrecognized."""
    if not raw:
        return None
    s = raw.strip().lower()
    for needle, value in _RULES:
        if needle in s:
            return value
    return None


def method_bucket(method: str | None) -> str | None:
    """Coarse bucket: 'ko' | 'decision' | 'draw' | 'dq' | 'nc' | None."""
    if not method:
        return None
    return _BUCKET.get(method)
