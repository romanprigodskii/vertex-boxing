"""No feature's missingness may name a corner.

The age group cost us a headline. age_a and age_b went missing independently of
each other, the profile crawl had gone after the fighters the odds feed quotes,
those men stand in the first corner, and the first corner wins 87% of the time —
so "we know his birthday" predicted "he won" at 0.86 against 0.14, on 68,095
bouts, through nothing but a NaN. Every number in the 31 July note was measured
on top of it.

The test is cheap and general: for every feature, split the bouts by whether it
is missing and compare A's win rate; and for every a_*/b_* pair, compare the two
one-sided-missing populations. A legitimate gap is allowed — a debutant has no
layoff and does lose more often — so this prints and ranks rather than fails,
and the judgement is whether the missingness is a fact about the fighter or a
fact about our crawl.

  ./venv/bin/python scripts/leak_check.py post-ingest
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
from src import features as F  # noqa: E402

CACHE = ROOT / "imports" / "staging"


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "post-ingest"
    df = pd.read_parquet(CACHE / f"sym_{tag}.parquet")
    feats = pd.read_parquet(CACHE / f"feats_{tag}_v{F.FEATS_VERSION}.parquet")
    y = F.label(df)
    nd = (~df["is_draw"]).to_numpy()
    y, feats = y[nd], feats[nd].reset_index(drop=True)
    base = y.mean()
    print(f"{len(y):,} decisive bouts · A wins {base:.4f} "
          f"(symmetrisation is working iff this is 0.50)\n")

    print("--- one-sided missing, per a_*/b_* pair "
          "(a gap here is a corner named by a NaN) ---")
    # A gap is not automatically a leak. A man having no previous bout is a
    # fact about the man, and debutants lose — so the test is not "is there a
    # gap" but "is the gap explained by his record or by our crawl". `debut`
    # is the share of the one-sided rows where that corner had never fought.
    n_a, n_b = feats["n_a"].to_numpy(), feats["n_b"].to_numpy()
    pairs = [(c, "b" + c[1:]) for c in feats.columns
             if c.startswith("a_") and "b" + c[1:] in feats.columns]
    pairs += [(c, c[:-2] + "_b") for c in feats.columns
              if c.endswith("_a") and c[:-2] + "_b" in feats.columns]
    worst = 0.0
    for a, b in sorted(set(pairs)):
        ka, kb = feats[a].notna().to_numpy(), feats[b].notna().to_numpy()
        oa, ob = ka & ~kb, ~ka & kb
        if oa.sum() < 50 or ob.sum() < 50:
            continue
        pa, pb = y[oa].mean(), y[ob].mean()
        # the missing side is B on the oa rows and A on the ob rows
        debut = np.concatenate([n_b[oa] == 0, n_a[ob] == 0]).mean()
        gap = abs(pa - pb)
        flag = ("  <-- LEAK" if gap > 0.15 and debut < 0.80
                else ("  (debut)" if gap > 0.05 else ""))
        if gap > 0.15 and debut < 0.80:
            worst = max(worst, gap)
        print(f"  {a:>12s}/{b:<12s} only-A n={oa.sum():7,} P={pa:.4f} | "
              f"only-B n={ob.sum():7,} P={pb:.4f} | Δ={pa - pb:+.4f} "
              f"· debut {debut:.0%}{flag}")
    print(f"  worst unexplained asymmetry: {worst:.4f}"
          f"{'  <-- FAIL' if worst > 0.15 else '  ok'}\n")

    print("--- missing vs present, per feature (legitimate when missingness is "
          "a fact about the fighter, not about the crawl) ---")
    rows = []
    for c in feats.columns:
        m = feats[c].isna().to_numpy()
        if m.mean() in (0.0, 1.0) or m.sum() < 50 or (~m).sum() < 50:
            continue
        rows.append((c, m.mean(), y[m].mean(), y[~m].mean()))
    for c, share, p1, p0 in sorted(rows, key=lambda r: -abs(r[2] - r[3]))[:14]:
        print(f"  {c:16s} missing {share:5.1%}  P(A|missing)={p1:.4f}  "
              f"P(A|present)={p0:.4f}  Δ={p1 - p0:+.4f}")

    print("\n--- single features that alone separate the corners too well ---")
    # A rating gap of 400 points SHOULD predict 97%, so the bar is set where no
    # honest pre-fight number can reach: near-determinism on both tails.
    for c in feats.columns:
        v = feats[c].to_numpy(float)
        ok = np.isfinite(v)
        if ok.sum() < 1000:
            continue
        q = np.nanquantile(v[ok], [0.02, 0.98])
        if q[0] == q[1]:
            continue
        hi, lo = v > q[1], v < q[0]
        if hi.sum() < 100 or lo.sum() < 100:
            continue
        sep = abs(y[hi].mean() - y[lo].mean())
        if sep > 0.975:
            print(f"  {c:16s} tails separate A's win rate by {sep:.3f} "
                  f"({y[lo].mean():.3f} → {y[hi].mean():.3f})")


if __name__ == "__main__":
    main()
