"""Freeze the corpus to a parquet so a series of measurements is comparable.

The DB grows under our feet — home_bias reported 21,310 bouts one day and
22,551 the next on "the same data", which makes two commits' numbers
incomparable. Every experiment reads a snapshot with a tag instead.

  ./venv/bin/python scripts/snapshot_corpus.py pre-ingest
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "scraper"))
from src.db import get_connection  # noqa: E402

OUT = ROOT / "imports" / "staging"

SQL = """
select e.date::date          as dt,
       b.fighter_a_id        as a,
       b.fighter_b_id        as b,
       fa.name_en            as a_name,
       fb.name_en            as b_name,
       b.winner_id, b.is_draw,
       b.method::text        as method,
       b.scheduled_rounds    as sched,
       b.weight_class::text  as div,
       b.round_finished,
       e.location_country    as country,
       e.location_city       as city,
       e.promoter,
       e.slug                as event_slug,
       -- what BoxRec printed on the night: the only pre-fight numbers that
       -- cover the career BEFORE our corpus starts
       b.a_bouts_before, b.b_bouts_before,
       b.a_wins_before, b.a_losses_before, b.a_draws_before,
       b.b_wins_before, b.b_losses_before, b.b_draws_before,
       fa.dob                as a_dob,
       fb.dob                as b_dob,
       fa.height_cm          as a_height,
       fb.height_cm          as b_height,
       fa.stance::text       as a_stance,
       fb.stance::text       as b_stance
from bout b
join event e   on e.id  = b.event_id
join fighter fa on fa.id = b.fighter_a_id
join fighter fb on fb.id = b.fighter_b_id
where b.status = 'completed' and e.date is not null
  and (b.winner_id is not null or b.is_draw)
order by e.date, b.id
"""


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    conn = get_connection()
    df = pd.read_sql(SQL, conn)
    conn.close()
    df["dt"] = pd.to_datetime(df["dt"])
    df = df[(df["dt"] >= "1950-01-01") & (df["dt"] <= pd.Timestamp.today())]
    for c in ("a", "b", "winner_id"):
        df[c] = df[c].astype(str)
    path = OUT / f"corpus_{tag}.parquet"
    df.reset_index(drop=True).to_parquet(path, index=False)
    day1 = (df["dt"].dt.day == 1).mean()
    print(f"{len(df):,} bouts → {path.name}")
    print(f"  {df['dt'].min().date()} → {df['dt'].max().date()} · "
          f"{day1:.1%} dated the 1st · "
          f"{df['sched'].notna().mean():.1%} with a distance · "
          f"{df['a_losses_before'].notna().mean():.1%} with the record split")


if __name__ == "__main__":
    main()
