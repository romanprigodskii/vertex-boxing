"""Static config for the boxing ingestion pipeline — endpoints, etiquette."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env.local")

CONTACT_EMAIL = os.environ.get("SCRAPER_CONTACT_EMAIL", "contact@example.com")

# Wikimedia/DBpedia etiquette asks for an identifying UA with a contact.
USER_AGENT = f"VertexBoxing/0.1 (ingest; {CONTACT_EMAIL})"

# Polite defaults. The open SPARQL/MediaWiki endpoints tolerate more than a
# scraped site, but we stay gentle — one request at a time, spaced out.
RATE_LIMIT_SECONDS = 1.0
REQUEST_TIMEOUT = 60.0
MAX_RETRIES = 4
RETRY_BACKOFF_BASE = 2.0

# --- Open-source endpoints (the clean bootstrap) ---
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
DBPEDIA_SPARQL = "https://dbpedia.org/sparql"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# boxing-data.com via RapidAPI (fighters / events / schedule / rankings).
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")
BOXING_DATA_HOST = "boxing-data-api.p.rapidapi.com"
BOXING_DATA_BASE = f"https://{BOXING_DATA_HOST}"

# Kaggle bootstrap dumps (prototype only — stale, BoxRec-provenance).
KAGGLE_DATASETS = [
    "iyadelwy/boxing-matches-dataset-predict-winner",
    "mexwell/boxing-matches",
]

# Wikidata property for the BoxRec boxer-ID crosswalk.
WIKIDATA_BOXREC_PROP = "P1967"
