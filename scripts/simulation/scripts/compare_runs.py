"""Paired bootstrap between two saved runs, on the bouts they share.

Two log-losses each carrying ±0.03 tell you nothing about which model is
better. The same bouts, differenced per bout, tell you everything.

  ./venv/bin/python scripts/compare_runs.py baseline-pre all
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PRED = Path(__file__).resolve().parents[3] / "imports" / "staging" / "preds"


def ll(p, y):
    return -np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-9, 1))


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: compare_runs.py <label_a> <label_b>")
        print("saved:", ", ".join(sorted(p.stem for p in PRED.glob("*.npz"))))
        return
    a, b = (np.load(PRED / f"{x}.npz", allow_pickle=True) for x in sys.argv[1:3])
    suff = "_corp" if "--corpus" in sys.argv else ""
    ka, kb = a["key" + suff], b["key" + suff]
    common = np.intersect1d(ka, kb)
    ia = {k: i for i, k in enumerate(ka)}
    ib = {k: i for i, k in enumerate(kb)}
    sa = np.array([ia[k] for k in common])
    sb = np.array([ib[k] for k in common])
    y = a["y" + suff][sa]
    assert (y == b["y" + suff][sb]).all(), "the two runs disagree about who won"
    la, lb = ll(a["p" + suff][sa], y), ll(b["p" + suff][sb], y)
    d = la - lb                      # positive = B is better
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"общих боёв: {len(common):,}")
    print(f"  {sys.argv[1]:22s} log-loss {la.mean():.4f}")
    print(f"  {sys.argv[2]:22s} log-loss {lb.mean():.4f}")
    print(f"  выигрыш B: {d.mean():+.4f} · 95% [{lo:+.4f}, {hi:+.4f}] · "
          + ("значимо" if lo > 0 else ("хуже" if hi < 0 else "неотличимо от нуля")))


if __name__ == "__main__":
    main()
