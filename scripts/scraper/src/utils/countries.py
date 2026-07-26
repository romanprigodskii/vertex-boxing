"""Country name → ISO-3166 alpha-2, for the names BoxRec actually prints.

`event.location_country` is char(2), so the page's "United Kingdom" has to be
folded to a code before it can be stored. Only the boxing world is covered —
unknown names return None rather than a guess, so a miss stays visible as NULL.
"""

from __future__ import annotations

_ISO2 = {
    "usa": "US", "united states": "US", "united states of america": "US",
    "mexico": "MX", "méxico": "MX", "argentina": "AR", "brazil": "BR", "colombia": "CO",
    "venezuela": "VE", "chile": "CL", "peru": "PE", "ecuador": "EC", "uruguay": "UY",
    "panama": "PA", "nicaragua": "NI", "costa rica": "CR", "dominican republic": "DO",
    "puerto rico": "PR", "cuba": "CU", "guatemala": "GT", "honduras": "HN",
    "el salvador": "SV", "bolivia": "BO", "paraguay": "PY", "canada": "CA",
    "united kingdom": "GB", "england": "GB", "scotland": "GB", "wales": "GB",
    "northern ireland": "GB", "ireland": "IE", "france": "FR", "germany": "DE",
    "spain": "ES", "italy": "IT", "portugal": "PT", "netherlands": "NL",
    "belgium": "BE", "switzerland": "CH", "austria": "AT", "denmark": "DK",
    "sweden": "SE", "norway": "NO", "finland": "FI", "iceland": "IS", "poland": "PL",
    "czech republic": "CZ", "czechia": "CZ", "slovakia": "SK", "hungary": "HU",
    "romania": "RO", "bulgaria": "BG", "serbia": "RS", "croatia": "HR",
    "slovenia": "SI", "bosnia and herzegovina": "BA", "montenegro": "ME",
    "north macedonia": "MK", "albania": "AL", "greece": "GR", "turkey": "TR",
    "russia": "RU", "ukraine": "UA", "belarus": "BY", "kazakhstan": "KZ",
    "uzbekistan": "UZ", "kyrgyzstan": "KG", "azerbaijan": "AZ", "armenia": "AM",
    "georgia": "GE", "moldova": "MD", "lithuania": "LT", "latvia": "LV",
    "estonia": "EE", "japan": "JP", "south korea": "KR", "korea": "KR",
    "north korea": "KP", "china": "CN", "hong kong": "HK", "taiwan": "TW",
    "thailand": "TH", "philippines": "PH", "indonesia": "ID", "malaysia": "MY",
    "singapore": "SG", "vietnam": "VN", "cambodia": "KH", "india": "IN",
    "pakistan": "PK", "bangladesh": "BD", "sri lanka": "LK", "nepal": "NP",
    "australia": "AU", "new zealand": "NZ", "fiji": "FJ", "papua new guinea": "PG",
    "south africa": "ZA", "ghana": "GH", "nigeria": "NG", "kenya": "KE",
    "tanzania": "TZ", "uganda": "UG", "cameroon": "CM", "ivory coast": "CI",
    "côte d'ivoire": "CI", "senegal": "SN", "morocco": "MA", "tunisia": "TN",
    "algeria": "DZ", "egypt": "EG", "namibia": "NA", "zimbabwe": "ZW",
    "zambia": "ZM", "botswana": "BW", "mozambique": "MZ", "congo": "CG",
    "dr congo": "CD", "united arab emirates": "AE", "saudi arabia": "SA",
    "qatar": "QA", "bahrain": "BH", "kuwait": "KW", "oman": "OM", "israel": "IL",
    "lebanon": "LB", "jordan": "JO", "iran": "IR", "iraq": "IQ", "armenia ": "AM",
    "jamaica": "JM", "trinidad and tobago": "TT", "bahamas": "BS", "barbados": "BB",
    "guyana": "GY", "suriname": "SR", "haiti": "HT", "curacao": "CW",
    "malawi": "MW", "afghanistan": "AF", "mauritius": "MU", "turkiye": "TR",
    "türkiye": "TR", "french polynesia": "PF", "french guiana": "GF",
    "new caledonia": "NC", "guadeloupe": "GP", "martinique": "MQ",
    "reunion": "RE", "réunion": "RE",
    "monaco": "MC", "malta": "MT", "cyprus": "CY", "luxembourg": "LU",
}


def iso2(name: str | None) -> str | None:
    if not name:
        return None
    return _ISO2.get(name.strip().lower())
