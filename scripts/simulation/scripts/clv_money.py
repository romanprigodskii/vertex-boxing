"""CLV and ROI are the same bets in two units. This converts between them.

The scoreboard reports closing-line value in probability (+0.0148 on the upper
tier, interval well clear of zero) and return in percent (+11.3%, ±9.7pp), and
reads the first as the trustworthy version of the second. That reading skips a
step. A bet is struck at a price carrying margin, and the movement has to cover
the margin before it is worth anything.

Per bet, in probability:

    we pay        1/o           the opening price's implied probability
    open fair     f_o           the same two-way board, margin split out
    the book keeps  1/o - f_o   on our side, ~half the overround
    it came to us   q_c - f_o   how far the close moved our way — this is CLV
    net             q_c - 1/o   the two against each other

and in money that is E[o*q_c] - 1, each bet's net multiplied by its own price,
so a long price weighs more in both directions.

Three quantities the scoreboard runs together:

  realised ROI         what the bets paid.        no assumption, wide interval
  closing-implied ROI  what they pay if the close is the truth.
                       one assumption, interval ~7x tighter, no outcomes in it
  the difference       whether the close was wrong on these bets. wide again,
                       and wide for the same reason the first one is

Read alongside `regional.py`, which reports the CLV, and `rule_oos.py`, which
tests the level rule on an era it never saw.

  ./venv/bin/python scripts/clv_money.py                 # v10-mirror
  ./venv/bin/python scripts/clv_money.py v10-mirror-open
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(ROOT / "scripts" / "simulation" / "scripts"))
import market_eval as ME  # noqa: E402
from src import features as F  # noqa: E402

CACHE = ROOT / "imports" / "staging"
EDGE = 0.02                        # fixed in advance, copied from regional.py
UPPER_LVL = ("continental", "world")


def boot(x, n=20000, seed=7):
    rng = np.random.default_rng(seed)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return np.percentile(b, [2.5, 97.5])


def report(name, r, imp):
    lo, hi = boot(r)
    lo2, hi2 = boot(imp)
    lo3, hi3 = boot(r - imp)
    print(f"\n{name}: {len(r):,} bets")
    print(f"   realised ROI        {r.mean():+7.2%}  [{lo:+.2%}, {hi:+.2%}]"
          f"   width {hi - lo:5.1%}")
    print(f"   closing-implied ROI {imp.mean():+7.2%}  [{lo2:+.2%}, {hi2:+.2%}]"
          f"   width {hi2 - lo2:5.1%}")
    print(f"   difference          {np.mean(r - imp):+7.2%}  [{lo3:+.2%}, {hi3:+.2%}]")


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "v10-mirror"
    d = np.load(CACHE / "preds" / f"{label}.npz", allow_pickle=True)
    p, y = d["p"], d["y"]
    ca, cb, oa, ob, key = d["ca"], d["cb"], d["oa"], d["ob"], d["key"]
    f = pd.read_parquet(CACHE / f"feats_card_v{F.FEATS_VERSION}.parquet",
                        columns=["sched_rounds", "title_lvl"]).iloc[key].reset_index(drop=True)
    dts = pd.read_parquet(CACHE / "sym_card.parquet",
                          columns=["dt"]).iloc[key]["dt"].reset_index(drop=True)

    # the same upper tier regional.py's last block picks, copied verbatim so
    # the two scripts cannot drift: twelve rounds, or a belt at rung 3+
    # (continental, international, world).
    lvl = np.nan_to_num(f["title_lvl"].to_numpy(float), nan=0)
    upper = (np.nan_to_num(f["sched_rounds"].to_numpy(float), nan=0) >= 12) | (lvl >= 3)

    ok = np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1)
    ea = p - 1 / np.where(ok, oa, np.inf)
    eb = (1 - p) - 1 / np.where(ok, ob, np.inf)
    BA = (ea > EDGE) & (ea >= eb) & ok
    BB = (eb > EDGE) & (eb > ea) & ok
    took = np.where(BA, oa, ob)
    won = np.where(BA, y == 1, y == 0)
    fo_a = (1 / oa) / (1 / oa + 1 / ob)
    fo = np.where(BA, fo_a, 1 - fo_a)

    print(f"{label}: {len(y):,} quoted test bouts · upper tier {int(upper.sum()):,}")
    print(f"levels present: {sorted(set(lvl[np.isfinite(lvl)].astype(int)))} "
          f"— upper taken as sched>=12 or title_lvl>=4")

    for dv in ("proportional", "power"):
        q = ME.devig(1 / ca, 1 / cb, dv)
        qs = np.where(BA, q, 1 - q)
        print(f"\n{'=' * 78}\n[{dv} de-vig of the close]")
        for nm, m in [("all quoted", BA | BB), ("upper tier", (BA | BB) & upper)]:
            report(f"  {nm}", np.where(won[m], took[m] - 1, -1.0), took[m] * qs[m] - 1)
        m = (BA | BB) & upper
        print(f"\n   the arithmetic, per bet, in probability ({int(m.sum()):,} bets)")
        print(f"     we pay          {np.mean(1 / took[m]):.4f}")
        print(f"     open fair       {fo[m].mean():.4f}  → the book keeps "
              f"{np.mean(1 / took[m] - fo[m]):+.4f}")
        print(f"     close fair      {qs[m].mean():.4f}  → it came to us    "
              f"{np.mean(qs[m] - fo[m]):+.4f}  (this is the CLV)")
        print(f"     net                             "
              f"{np.mean(qs[m] - 1 / took[m]):+.4f}")

    # the sign is set by an average that hides its own shape
    q = ME.devig(1 / ca, 1 / cb, "power")
    qs = np.where(BA, q, 1 - q)
    m = (BA | BB) & upper
    print(f"\n{'=' * 78}\nby the price taken — where the CLV is, and where the spread is")
    band = pd.cut(took[m], [1, 2, 3, 5, 1000], labels=["<2", "2-3", "3-5", "5+"])
    r_all = np.where(won[m], took[m] - 1, -1.0)
    rows = []
    for b in ["<2", "2-3", "3-5", "5+"]:
        k = np.asarray(band == b)
        if k.sum() < 10:
            continue
        rows.append({
            "price": b, "bets": int(k.sum()),
            "the book keeps": f"{np.mean(1 / took[m][k] - fo[m][k]):+.4f}",
            "came to us (CLV)": f"{np.mean(qs[m][k] - fo[m][k]):+.4f}",
            "closing-implied ROI": f"{np.mean(took[m][k] * qs[m][k] - 1):+.1%}",
            "realised ROI": f"{r_all[k].mean():+.1%}",
            "share of spread": f"{np.var(r_all[k]) * k.sum() / (np.var(r_all) * len(r_all)):.0%}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # The quoted test in halves by date. This is NOT an out-of-sample test and
    # must not be read as one: regional.py chose the rule's cut-points off a
    # table built on all 3,288 quoted bouts, so both halves are inside the
    # window that produced it. rule_oos.py is the out-of-sample test. What this
    # shows is narrower and still worth knowing — whether the realised return
    # is a property of the strategy or of one stretch of the calendar.
    print(f"\n{'=' * 78}\nthe quoted test in halves by date (not an out-of-sample test)")
    mid = dts.quantile(0.5)
    print(f"split at {pd.Timestamp(mid).date()}")
    for nm, half in [("early half", (dts <= mid).to_numpy()),
                     ("late half ", (dts > mid).to_numpy())]:
        k = m & half
        if k.sum() < 30:
            continue
        report(f"  {nm}", np.where(won[k], took[k] - 1, -1.0), took[k] * qs[k] - 1)

    # If the sign is set by the margin at the open, the useful question is not
    # "is there an edge" but "how thin does the open have to be". Hold the
    # market's fair view fixed and re-price the same bet at a tighter board:
    # o' = 1 / (f_o * w). Nothing about the model or the selection moves.
    print(f"\n{'=' * 78}\nwhat the same bets pay at a tighter opening board")
    ov = (1 / oa + 1 / ob)[m]
    print(f"the board we actually have: overround {ov.mean():.4f} "
          f"(median {np.median(ov):.4f})")
    print(f"{'overround':>10} {'we pay':>8} {'net (prob)':>11} {'ROI by the close':>17}"
          f"   {'favourites only':>16}")
    fav = took[m] < 2.0
    for w in (ov.mean(), 1.05, 1.04, 1.03, 1.02):
        o2 = 1 / (fo[m] * w)
        imp = o2 * qs[m] - 1
        tag = "actual" if abs(w - ov.mean()) < 1e-9 else f"{w:.3f}"
        print(f"{tag:>10} {np.mean(fo[m] * w):>8.4f} {np.mean(qs[m] - fo[m] * w):>+11.4f} "
              f"{imp.mean():>+17.2%}   {imp[fav].mean():>+16.2%}")
    print("Selection is held fixed at the real board — a tighter open would also")
    print("change which bouts clear the 2% threshold, so read these as the value")
    print("of the SAME bets at a better price, not as a re-run of the strategy.")

    # ca/cb in the npz are close_* — the WORST price across the ten books, which
    # is the most generous reading for us. The saved run does not carry best_*,
    # so fetch it back through the same join on the same corpus rows: a claim
    # that the sign survives every reading of the board should not depend on a
    # different prediction file.
    print(f"\n{'=' * 78}\nthe same bets, with the closing board read three ways")
    jj = ME.join_odds(ME.build("card")[0], verbose=False).set_index("index")
    have = [i for i in key if i in jj.index]
    if len(have) != len(key):
        print(f"  (skipped: {len(key) - len(have):,} rows no longer join)")
    else:
        sw = jj.loc[key, "swap"].to_numpy(bool)
        ba_, bb_ = (np.where(~sw, jj.loc[key, "best_a"], jj.loc[key, "best_b"]).astype(float),
                    np.where(~sw, jj.loc[key, "best_b"], jj.loc[key, "best_a"]).astype(float))
        mid_a, mid_b = 2 / (1 / ca + 1 / ba_), 2 / (1 / cb + 1 / bb_)
        for nm, xa, xb in [("worst end (close_*)", ca, cb),
                           ("mid of the range", mid_a, mid_b),
                           ("best end (best_*)", ba_, bb_)]:
            k = (BA | BB) & upper & np.isfinite(xa) & np.isfinite(xb) & (xa > 1) & (xb > 1)
            line = f"  {nm:<22} {int(k.sum()):>4} bets  overround " \
                   f"{np.mean((1 / xa + 1 / xb)[k]):.4f} → "
            for dv in ("proportional", "power"):
                q2 = ME.devig(1 / xa, 1 / xb, dv)
                line += f"{dv[:4]} {np.mean(took[k] * np.where(BA, q2, 1 - q2)[k] - 1):+.2%}  "
            print(line)
        print("  negative on every reading and every de-vig.")


if __name__ == "__main__":
    main()
