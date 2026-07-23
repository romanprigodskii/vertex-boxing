"""Normalize the many boxing-division spellings to our 17 enum values.

Boxing divisions carry two naming systems — the "super/light" American style
and the "junior/light" British style — plus poundage and abbreviations. Every
source (Wikidata, DBpedia, Wikipedia tables, boxing-data.com, odds feeds)
uses a slightly different one, so ingest funnels all of them through here.
"""

from __future__ import annotations

import re

# Canonical enum value → every synonym we might see (lowercased).
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "minimumweight": ("minimumweight", "strawweight", "mini flyweight", "minimum", "105"),
    "light_flyweight": ("light flyweight", "junior flyweight", "108"),
    "flyweight": ("flyweight", "112"),
    "super_flyweight": ("super flyweight", "junior bantamweight", "115"),
    "bantamweight": ("bantamweight", "118"),
    "super_bantamweight": ("super bantamweight", "junior featherweight", "122"),
    "featherweight": ("featherweight", "126"),
    "super_featherweight": ("super featherweight", "junior lightweight", "130"),
    "lightweight": ("lightweight", "135"),
    "super_lightweight": ("super lightweight", "junior welterweight", "light welterweight", "140"),
    "welterweight": ("welterweight", "147"),
    "super_welterweight": (
        "super welterweight",
        "junior middleweight",
        "light middleweight",
        "154",
    ),
    "middleweight": ("middleweight", "160"),
    "super_middleweight": ("super middleweight", "168"),
    "light_heavyweight": ("light heavyweight", "175"),
    "cruiserweight": ("cruiserweight", "junior heavyweight", "200"),
    "heavyweight": ("heavyweight",),
}

# Longest synonyms first so "super flyweight" wins over "flyweight".
_LOOKUP: list[tuple[str, str]] = sorted(
    ((syn, canon) for canon, syns in _SYNONYMS.items() for syn in syns),
    key=lambda kv: len(kv[0]),
    reverse=True,
)


def normalize_division(raw: str | None) -> str:
    """Return a weight_class enum value; 'catchweight' for stated catchweights,
    'unknown' when nothing matches."""
    if not raw:
        return "unknown"
    s = raw.strip().lower()
    s = s.replace("weight class", "").replace("division", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s).strip()
    if "catch" in s:
        return "catchweight"
    for syn, canon in _LOOKUP:
        if syn in s:
            return canon
    return "unknown"
