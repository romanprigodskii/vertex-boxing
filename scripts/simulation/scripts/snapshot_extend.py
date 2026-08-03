"""Add newly-parsed pre-bell columns to an EXISTING corpus snapshot.

snapshot_corpus.py rebuilds from the database, and the database grows: it holds
415,283 bouts today against the 413,279 in `corpus_card`, so re-running it to
pick up two new columns would also silently move the population every published
number was measured on. A snapshot exists to stop that happening.

This takes the frozen snapshot as it is, joins columns that are derived from
files already on disk, and writes a new tag with the SAME rows in the SAME
order. Anything that was measurable before is measurable after, and the two
tags are comparable bout for bout.

  ./venv/bin/python scripts/snapshot_extend.py --from card --to l6
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STAGING = ROOT / "imports" / "staging"

# what to bring over from event_extras.parquet, and under what name
EXTRA = {"a_l6": "a_l6", "b_l6": "b_l6", "a_rec": "a_rec", "b_rec": "b_rec"}


def arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    src, dst = arg("--from", "card"), arg("--to", "l6")
    base = pd.read_parquet(STAGING / f"corpus_{src}.parquet")
    ex = pd.read_parquet(STAGING / "event_extras.parquet")
    n0, cols0 = len(base), list(base.columns)

    ex = ex[["boxrec_id", *EXTRA]].rename(columns=EXTRA)
    ex["boxrec_id"] = ex["boxrec_id"].astype(str)
    # one row per bout id, or the merge multiplies the corpus
    ex = ex.drop_duplicates("boxrec_id", keep="first")
    key = base["bout_bxid"].astype(str)
    out = base.merge(ex, left_on=key, right_on="boxrec_id", how="left") \
              .drop(columns=["boxrec_id", "key_0"], errors="ignore")
    assert len(out) == n0, f"the join changed the population: {len(out)} vs {n0}"
    assert list(out.columns)[:len(cols0)] == cols0, "the join reordered the base columns"

    path = STAGING / f"corpus_{dst}.parquet"
    out.reset_index(drop=True).to_parquet(path, index=False)
    both = out["a_l6"].notna() & out["b_l6"].notna()
    print(f"{len(out):,} bouts → {path.name}")
    print(f"  a form strip on both corners {both.mean():.1%} · "
          f"on at least one {(out['a_l6'].notna() | out['b_l6'].notna()).mean():.1%}")
    print(f"  the page was parsed at all on {out['card_n'].notna().mean():.1%}")


if __name__ == "__main__":
    main()
