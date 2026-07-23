"""psycopg connection — reads DATABASE_URL from the project-root .env.local,
verbatim from the vertexmma scraper (same Supabase keepalive tuning)."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# scripts/scraper/src/db.py → project root is 3 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env.local")

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Populate .env.local at the project root.")


def get_connection() -> psycopg.Connection:
    # autocommit=False so callers can batch inserts in transactions.
    # TCP keepalives keep the Supabase pooler from killing an idle socket
    # during long SPARQL/paginated pulls.
    return psycopg.connect(
        DATABASE_URL,
        autocommit=False,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
