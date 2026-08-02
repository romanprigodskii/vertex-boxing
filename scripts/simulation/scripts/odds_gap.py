"""What is missing a price, and what would finding it buy.

Section 7 of the status doc says the bottleneck is not the model but the width
of the ROI interval, and the width is set by how many upper-tier bouts carry a
quote. This turns that sentence into a work order: it counts the bouts that
*should* have a price and do not, and writes them out as a target list any new
odds source can be scored against.

Two numbers matter and they are different:
  - coverage  — of the bouts in the corpus, which have a quote at all
  - shortfall — how many more quoted upper-tier bouts the ROI interval needs

Usage:
  odds_gap.py                     # the map: coverage by year and by level
  odds_gap.py --dump targets.csv  # the unpriced upper-tier bouts, for hunting
  odds_gap.py --power             # what N buys, given the observed ROI spread
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation" / "scripts"))
import market_eval as ME  # noqa: E402

CORPUS = ROOT / "imports" / "staging" / "corpus_card.parquet"

# The upper tier, as regional.py's last block defines it: scheduled twelve, or
# a belt at rung 3 or above. The prose in the status doc says "continental or
# world" and the code says `title_lvl >= 3`, which also takes in the
# international belts — the code is what produced the reported numbers, so the
# code is what this copies.
UPPER_BELTS = ("continental", "international", "world")
CUTOFF = pd.Timestamp("2023-06-01")  # model cut-off; everything before is clean


def upper_mask(d: pd.DataFrame) -> np.ndarray:
    return ((d["sched"].fillna(0) >= 12) | d["title_level"].isin(UPPER_BELTS)).to_numpy()


def main() -> None:
    d = pd.read_parquet(CORPUS)
    d["dt"] = pd.to_datetime(d["dt"])
    d = d[~d["is_draw"]].reset_index(drop=True)

    j = ME.join_odds(d, verbose=True)
    priced = set(zip(j["dt"], j["a_name"], j["b_name"]))
    d["priced"] = [k in priced for k in zip(d["dt"], d["a_name"], d["b_name"])]
    d["upper"] = upper_mask(d)
    d["titled"] = d["title_level"].notna()

    win = (d["dt"] >= "2015-01-01") & (d["dt"] < CUTOFF)

    print("\n=== coverage, 2015-01 → 2023-06 (the clean window) ===")
    rows = []
    for label, m in [
        ("all bouts", win),
        ("any belt", win & d["titled"]),
        ("scheduled 12", win & (d["sched"].fillna(0) >= 12)),
        ("continental/world belt", win & d["title_level"].isin(UPPER_BELTS)),
        ("UPPER TIER (12r or cont./world)", win & d["upper"]),
    ]:
        n, p = int(m.sum()), int((m & d["priced"]).sum())
        rows.append({"slice": label, "bouts": n, "priced": p,
                     "coverage": f"{p / max(n, 1):.1%}", "unpriced": n - p})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== upper tier by year ===")
    u = d[win & d["upper"]].copy()
    u["year"] = u["dt"].dt.year
    by = u.groupby("year").agg(bouts=("priced", "size"), priced=("priced", "sum"))
    by["coverage"] = (by["priced"] / by["bouts"]).map("{:.1%}".format)
    by["unpriced"] = by["bouts"] - by["priced"]
    print(by.to_string())

    print("\n=== where the unpriced upper-tier bouts are (top 15 countries) ===")
    miss = u[~u["priced"]]
    print(miss["country"].fillna("?").value_counts().head(15).to_string())

    if "--power" in sys.argv:
        # The observed spread: +11.3% ROI with [+1.8%, +21.2%] on 798 bets is a
        # half-width of 9.7pp, so SE = 4.95pp and the per-bet SD is
        # 4.95 * sqrt(798) = 139.8% of a unit stake. Everything below follows
        # from that one number, and it is the honest one — it is measured on
        # this model's own bets, not assumed.
        sd = 0.0495 * np.sqrt(798)
        print(f"\n=== what N buys (per-bet SD = {sd:.3f} units, from the 798-bet run) ===")
        print(f"{'bets':>8} {'SE':>8} {'±95%':>8}   verdict at true ROI +4% / +11.3%")
        for n in (798, 1500, 2500, 4000, 6600, 10000, 20000, 30300):
            se = sd / np.sqrt(n)
            hw = 1.96 * se
            v4 = "excludes 0" if hw < 0.04 else "includes 0"
            v11 = "excludes 0" if hw < 0.113 else "includes 0"
            print(f"{n:>8,} {se:>7.2%} {hw:>7.2%}   {v4:>10} / {v11}")
        # The bet rate: 798 bets came out of how many quoted upper-tier bouts?
        q = int((win & d["upper"] & d["priced"]).sum())
        print(f"\nquoted upper-tier bouts in the window: {q:,}")
        print(f"unpriced upper-tier bouts in the window: {int((win & d['upper'] & ~d['priced']).sum()):,}")

    if "--dump" in sys.argv:
        out = ROOT / sys.argv[sys.argv.index("--dump") + 1]
        cols = ["dt", "a_name", "b_name", "sched", "title_level", "div",
                "country", "city", "event_slug", "card_n"]
        t = miss[cols].sort_values("dt")
        t.to_csv(out, index=False)
        print(f"\n{len(t):,} unpriced upper-tier bouts → {out}")


if __name__ == "__main__":
    main()
