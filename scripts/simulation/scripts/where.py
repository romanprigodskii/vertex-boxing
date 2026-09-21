"""Where the market's advantage actually sits, in OUR terms.

The headline says the model loses to the close by 0.033 and the band table says
where that sits on the PRICE's scale — which is useless for deciding what to
build, because we cannot see the price before the bell. This asks the same
question on axes we can see: how thin is our evidence, how experienced are the
two men, how big is the card. If the market's edge is concentrated where our
data is thin, then more data is the answer; if it is flat, the missing thing is
private information and no amount of crawling will find it.

  python3 scripts/where.py final-close --tag l6
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


def ll(p, y):
    return -np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-9, 1))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lab = args[0] if args else "final-close"
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "l6"
    args = [a for a in args if a != tag]
    d = np.load(CACHE / "preds" / f"{lab}.npz", allow_pickle=True)
    p, y, pm, key = d["p"], d["y"], d["p_mkt"], d["key"]
    want = ["rd_max", "n_min", "ntrue_min", "sched_rounds", "card_size", "d_bt8"]
    feats = pd.read_parquet(CACHE / F.cache_name("feats", tag), columns=want)
    f = feats.iloc[key].reset_index(drop=True)
    lm, lk = ll(p, y), ll(pm, y)
    print(f"{lab}: {len(y):,} quoted test bouts · model {lm.mean():.4f} · "
          f"market {lk.mean():.4f} · gap {(lm - lk).mean():+.4f}\n")

    def split(name, v, edges):
        v = np.asarray(v, float)
        print(f"--- {name} ---")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v >= lo) & (v < hi)
            if m.sum() < 40:
                continue
            print(f"  [{lo:8.4g},{hi:8.4g})  n={m.sum():5d}  model {lm[m].mean():.4f}  "
                  f"market {lk[m].mean():.4f}  gap {(lm[m] - lk[m]).mean():+.4f}  "
                  f"share of total gap {(lm[m] - lk[m]).sum() / (lm - lk).sum():6.1%}")
        nan = ~np.isfinite(v)
        if nan.sum() >= 40:
            print(f"  {'missing':>19s}  n={nan.sum():5d}  model {lm[nan].mean():.4f}  "
                  f"market {lk[nan].mean():.4f}  gap {(lm[nan] - lk[nan]).mean():+.4f}")

    q = lambda c, n=5: list(np.nanquantile(f[c].to_numpy(float),  # noqa: E731
                                           np.linspace(0, 1, n + 1)))
    split("rating deviation, the larger of the two (our evidence is thin →)",
          f["rd_max"], q("rd_max"))
    split("bouts behind the less experienced man", f["n_min"],
          [0, 1, 3, 6, 10, 20, 1e9])
    split("BoxRec bouts behind the less experienced man", f["ntrue_min"],
          [0, 3, 8, 15, 25, 40, 1e9])
    split("scheduled rounds", f["sched_rounds"], [0, 5, 7, 9, 11, 13])
    split("bouts on the card", f["card_size"], [0, 5, 8, 11, 15, 1e9])
    split("how far the whole-history rating is from even",
          f["d_bt8"].abs(), q("d_bt8"))
    split("our own confidence |logit|", np.abs(np.log(p / (1 - p))),
          [0, 0.5, 1, 1.5, 2.5, 4, 1e9])
    print("--- who is more confident ---")
    z = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))  # noqa: E731
    more = np.abs(z(p)) > np.abs(z(pm))
    for nm, m in (("model bolder", more), ("market bolder", ~more)):
        print(f"  {nm:14s} n={m.sum():5d}  model {lm[m].mean():.4f}  "
              f"market {lk[m].mean():.4f}  gap {(lm[m] - lk[m]).mean():+.4f}  "
              f"share {(lm[m] - lk[m]).sum() / (lm - lk).sum():6.1%}")
    print("\n  agreement on direction: "
          f"{(np.sign(z(p)) == np.sign(z(pm))).mean():.1%}")
    dis = np.sign(z(p)) != np.sign(z(pm))
    if dis.sum() > 30:
        print(f"  when they disagree on WHO wins (n={dis.sum()}): "
              f"model right {y[dis].mean() if True else 0:.3f} of the time by A's "
              f"outcome — model {lm[dis].mean():.4f} vs market {lk[dis].mean():.4f}")


if __name__ == "__main__":
    main()
