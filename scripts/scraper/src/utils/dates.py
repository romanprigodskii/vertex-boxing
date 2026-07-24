"""Lenient date parsing for heterogeneous boxing sources."""

from __future__ import annotations

from datetime import date, datetime

_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d %B %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%b %d, %Y",
    "%b %Y",   # BoxRec anonymous view shows month-year for historical bouts
    "%B %Y",
    "%Y",
)


def parse_date(raw: str | None) -> date | None:
    """Best-effort parse of a date string to a `date`. Returns None on failure
    (a bout we can't place on the timeline is skipped, never guessed — the
    point-in-time guarantee depends on real dates)."""
    if not raw:
        return None
    raw = raw.strip()
    # ISO datetime (Wikidata / SPARQL) — take the date part.
    if "T" in raw:
        raw = raw.split("T", 1)[0]
    for fmt in _FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None
