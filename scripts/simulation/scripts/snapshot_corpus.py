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
       -- The scales, on 81% of the corpus and over 90% of recent years. These
       -- were kept out of the model as "post-hoc data", which is right for a
       -- forecast made a month out and wrong for the test we actually run: the
       -- weigh-in is the day before, so the CLOSING line has already seen the
       -- man who came in four pounds heavy. Fair against the close, NOT fair
       -- against the open — the WEIGH group exists so the difference can be
       -- reported rather than assumed.
       b.a_weight_lbs::float8 as a_lbs,
       b.b_weight_lbs::float8 as b_lbs,
       -- The judges' totals, on 144,337 bouts. To Elo a win is a win, so
       -- 120-108 and 115-113 are the same evidence; they are not. The cards say
       -- by HOW MUCH, which is the one graded observation this sport hands out
       -- for free. Post-fight, so they may only ever update a rating for the
       -- NEXT bout — never appear as a feature of the bout they came from.
       case when jsonb_typeof(b.judges) = 'array' then
         (select avg((j->>'a')::numeric)::float8 from jsonb_array_elements(b.judges) j
          where j ? 'a') end as a_score,
       case when jsonb_typeof(b.judges) = 'array' then
         (select avg((j->>'b')::numeric)::float8 from jsonb_array_elements(b.judges) j
          where j ? 'b') end as b_score,
       -- The officials. A referee's taste for an early stoppage correlates 0.76
       -- between the first and second halves of his own career, on 1,235 men
       -- with 60 bouts or more — that is a personality, not noise, and the model
       -- has never seen it. Judges come with their individual cards, so each
       -- one's leaning is measurable on his own opinion rather than on the
       -- verdict he was outvoted into. Assigned by the commission before the
       -- bell and printed on the card, so fair against a closing line.
       b.referee_boxrec_id as ref_id,
       case when jsonb_typeof(b.judges) = 'array' then
         (select string_agg(j->>'boxrec_id', ',' order by o)
          from jsonb_array_elements(b.judges) with ordinality t(j, o)
          where j ? 'a' and j ? 'b') end as judge_ids,
       case when jsonb_typeof(b.judges) = 'array' then
         (select string_agg(j->>'a', ',' order by o)
          from jsonb_array_elements(b.judges) with ordinality t(j, o)
          where j ? 'a' and j ? 'b') end as judge_a,
       case when jsonb_typeof(b.judges) = 'array' then
         (select string_agg(j->>'b', ',' order by o)
          from jsonb_array_elements(b.judges) with ordinality t(j, o)
          where j ? 'a' and j ? 'b') end as judge_b,
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
