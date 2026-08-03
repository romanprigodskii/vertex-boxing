"""The knobs on the RATINGS, which have never been turned.

`ELO_K = 32` is Chess Federation's number for a club player, and it has sat in
features.py since the first commit. So has Glicko's `c` (50 rating points of
uncertainty per idle year), its `tau`, the initial RD of 350, the slow Elo's
K of 12, and Bradley-Terry's two decay constants. A 500-trial TPE search has
been run on this problem and found nothing — but it searched LightGBM, and a
badly set K is not noise a tree can average away. It is a systematic distortion
of the single strongest feature in the matrix, identical on every row, and the
model has no way to invert it.

This screens them the cheap way: a rating's own forecast, scored by log-loss,
with no boosting anywhere in the loop. That is a proxy — a rating can forecast
worse alone and still be a better feature, because the trees see the gap
alongside twenty other things — so nothing here is a result. It is a shortlist,
and the shortlist is then confirmed on the bench, three seeds, paired bootstrap.

The evaluation window ends at the corpus cutoff, so the reporting holdout is
never touched: choosing K on the same bouts that report it is the selection tax
tune.py exists to avoid.

  ./venv/bin/python scripts/rating_scan.py --what elo
  ./venv/bin/python scripts/rating_scan.py --what glicko --tag l6
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
from src import features as F  # noqa: E402

# the corpus holdout begins here; everything below is measured strictly before
CUTOFF = "2023-06-10"
EVAL_FROM = "2015-01-01"


def arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(np.where(y == 1, np.log(p), np.log1p(-p))))


class Corpus:
    """One pass over the parquet, reused by every configuration."""

    def __init__(self, tag: str) -> None:
        df = F.load(tag)
        self.a = df["a"].astype(str).to_numpy()
        self.b = df["b"].astype(str).to_numpy()
        w = df["winner_id"].astype(str).to_numpy()
        d = df["is_draw"].to_numpy(bool)
        self.sa = np.where(d, 0.5, (w == self.a).astype(float))
        self.draw = d
        self.dt = df["dt"].to_numpy("datetime64[D]")
        self.sched = pd.to_numeric(df["sched"], errors="coerce").to_numpy(float)
        meth = df["method"].astype(str).str.lower().to_numpy()
        self.stop = np.isin(meth, list(F.STOP))
        self.nover = np.isin(meth, list(F.NO_VERDICT))
        lo = np.datetime64(EVAL_FROM, "D")
        hi = np.datetime64(CUTOFF, "D")
        self.win = (self.dt > lo) & (self.dt <= hi) & ~d
        print(f"{len(self.a):,} bouts · scoring window {EVAL_FROM}..{CUTOFF} "
              f"= {int(self.win.sum()):,} decided bouts", flush=True)

    def score(self, p: np.ndarray, n_a: np.ndarray, n_b: np.ndarray) -> dict:
        m = self.win & np.isfinite(p)
        prem = m & (np.nan_to_num(self.sched, nan=0) >= 8) \
                 & (np.minimum(n_a, n_b) >= 8)
        y = (self.sa == 1.0).astype(int)
        return {"n": int(m.sum()), "ll": ll(p[m], y[m]),
                "n_prem": int(prem.sum()), "ll_prem": ll(p[prem], y[prem])}


# ------------------------------------------------------------------------ Elo
def run_elo(C: Corpus, k: float, scale: float = 400.0, k_new: float | None = None,
            new_until: int = 10, mov: float = 1.0) -> np.ndarray:
    """`k_new` is a larger step for a fighter's first `new_until` bouts, the
    provisional rating every federation uses and this project never had; `mov`
    multiplies K on a stoppage, which is elo_mov's idea applied to the main
    rating rather than to a second copy of it."""
    r = defaultdict(lambda: F.ELO_INIT)
    n = defaultdict(int)
    p = np.empty(len(C.a))
    na = np.empty(len(C.a)); nb = np.empty(len(C.a))
    for i in range(len(C.a)):
        a, b = C.a[i], C.b[i]
        ra, rb = r[a], r[b]
        e = 1.0 / (1.0 + 10 ** ((rb - ra) / scale))
        p[i] = e
        na[i], nb[i] = n[a], n[b]
        s = C.sa[i]
        ka = k_new if (k_new is not None and n[a] < new_until) else k
        kb = k_new if (k_new is not None and n[b] < new_until) else k
        m = mov if C.stop[i] else (0.7 if C.nover[i] else 1.0)
        r[a] = ra + ka * m * (s - e)
        r[b] = rb + kb * m * ((1 - s) - (1 - e))
        n[a] += 1; n[b] += 1
    return p, na, nb


# --------------------------------------------------------------------- Glicko
def run_glicko(C: Corpus, c_year: float = 50.0, tau: float = 0.5,
               rd0: float = 350.0, cons: float = 0.0) -> np.ndarray:
    """`cons` reproduces the conservative reading the feature set uses
    (rating − cons·RD) so the shrinkage can be tuned rather than assumed at 2."""
    g = F.Glicko2()
    g.rd = defaultdict(lambda: rd0)
    F._TAU = tau                       # module-level in Glickman's own notation
    old_c = 50.0
    p = np.empty(len(C.a))
    na = np.empty(len(C.a)); nb = np.empty(len(C.a))
    n = defaultdict(int)

    def peek(f, dt):
        rd = g.rd[f]
        if f in g.last:
            days = max((dt - g.last[f]).days, 0)
            rd = min(math.sqrt(rd * rd + (c_year ** 2) * (days / 365.0)), rd0)
        return g.r[f], rd

    g.peek = peek                      # same object, a tunable inflation term
    for i in range(len(C.a)):
        a, b = C.a[i], C.b[i]
        dt = pd.Timestamp(C.dt[i])
        ra, rda = peek(a, dt)
        rb, rdb = peek(b, dt)
        ea = ra - cons * rda
        eb = rb - cons * rdb
        phi = math.sqrt(rda * rda + rdb * rdb) / F._Q
        p[i] = 1.0 / (1.0 + math.exp(-max(min(F._g(phi) * (ea - eb) / F._Q, 30.0), -30.0)))
        na[i], nb[i] = n[a], n[b]
        g.update(a, b, C.sa[i], dt)
        n[a] += 1; n[b] += 1
    F._TAU = 0.5
    del old_c
    return p, na, nb


# K and the logistic scale are not two knobs. The update moves the rating by
# K·(s−e) and the forecast reads (ra−rb)/scale, so multiplying both by the same
# factor leaves every probability unchanged: K=40/scale=400 and K=32/scale=320
# score identically to the fourth decimal, which is the check that this is
# implemented right. The family is one-dimensional in K/scale, and the grid is
# written in K at the conventional scale of 400.
GRIDS = {
    "elo": [("K=%g" % k, dict(k=k)) for k in (8, 12, 16, 24, 32, 40, 48, 56, 64,
                                              80, 96, 128)]
           + [("K=%g new=%g" % (k, kn), dict(k=k, k_new=kn))
              for k in (24, 32, 48, 64) for kn in (64.0, 96.0, 128.0)]
           + [("K=48 new=96 u=%d" % u, dict(k=48, k_new=96.0, new_until=u))
              for u in (4, 6, 15, 25)]
           + [("K=%g mov=%g" % (k, m), dict(k=k, mov=m))
              for k in (24, 48) for m in (1.25, 1.5)],
    "glicko": [("c=%g" % c, dict(c_year=c)) for c in (20, 35, 50, 80, 120)]
              + [("tau=%g" % t, dict(tau=t)) for t in (0.2, 0.35, 0.8)]
              + [("rd0=%g" % r, dict(rd0=r)) for r in (250.0, 300.0)]
              + [("cons=%g" % c, dict(cons=c)) for c in (0.5, 1.0, 2.0)],
}


def main() -> None:
    what = arg("--what", "elo")
    C = Corpus(arg("--tag", "card"))
    runner = {"elo": run_elo, "glicko": run_glicko}[what]
    rows = []
    for name, kw in GRIDS[what]:
        t0 = time.time()
        p, na, nb = runner(C, **kw)
        s = C.score(p, na, nb)
        s["name"] = name
        rows.append(s)
        print(f"  {name:16s} ll {s['ll']:.4f}  prem {s['ll_prem']:.4f} "
              f"(n {s['n']:,} / {s['n_prem']:,})  {time.time()-t0:.0f}s", flush=True)
    r = pd.DataFrame(rows).sort_values("ll")
    print("\nranked by log-loss of the rating's own forecast:")
    print(r[["name", "ll", "ll_prem", "n", "n_prem"]].to_string(index=False))
    best = r.iloc[0]
    print(f"\nbest: {best['name']}  ({best['ll']:.4f})  — a shortlist, not a result; "
          f"confirm on the bench before believing it")


if __name__ == "__main__":
    main()
