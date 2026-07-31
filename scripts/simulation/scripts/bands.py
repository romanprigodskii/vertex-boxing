"""Where the gap to the closing line actually is.

A single number — "the model loses by 0.0396" — hides the shape of the loss,
and the shape is the whole diagnosis. Split the test bouts by how far the price
sat from 50/50 and the failure stops looking like a shortage of information:

  0.00-0.05 from even   372 bouts   model 0.1041   market 0.0606   +0.0436
  0.10-0.20 from even   448 bouts   model 0.4517   market 0.4486   +0.0031
  0.30-0.50 from even   333 bouts   model 0.7683   market 0.6854   +0.0830

In the middle the model is the market's equal. The whole gap is made at the two
ends, and the errors point OPPOSITE ways — too bold on a pick'em (0.768 against
a coin flip's 0.693), too timid on a 95% favourite. One global calibration slope
cannot fix both, which is why fitting one never helped.

The bands are read off the price, so this is a diagnostic and never an input:
nothing the model uses may know what the market thought.

  ./venv/bin/python scripts/bands.py final r-regime
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PRED = Path(__file__).resolve().parents[3] / "imports" / "staging" / "preds"
EDGES = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50)]


def ll(q, y):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return -np.where(y == 1, np.log(q), np.log1p(-q))


def main() -> None:
    labels = sys.argv[1:] or ["final"]
    runs = [(x, np.load(PRED / f"{x}.npz", allow_pickle=True)) for x in labels]
    # two readings of the same board can keep different bouts — a row with no
    # best price survives one and not the other — so compare on what they share
    common = runs[0][1]["key"]
    for _, d in runs[1:]:
        common = np.intersect1d(common, d["key"])
    sel = []
    for lab, d in runs:
        ix = {k: i for i, k in enumerate(d["key"])}
        sel.append((lab, d, np.array([ix[k] for k in common])))
    runs = [(lab, {k: d[k][s] if d[k].shape[:1] == d["key"].shape else d[k]
                   for k in ("p", "y", "p_mkt")}) for lab, d, s in sel]
    y, pm = runs[0][1]["y"], runs[0][1]["p_mkt"]
    q = np.minimum(pm, 1 - pm)             # how close the market thought it was
    head = f"{'полоса от 50/50':>18s} {'n':>6s} {'рынок':>8s}"
    for lab, _ in runs:
        head += f" {lab[:10]:>10s} {'вклад':>8s}"
    print(head)
    for lo, hi in EDGES:
        m = (q >= lo) & (q < hi)
        if m.sum() < 10:
            continue
        row = f"{lo:.2f}-{hi:.2f}{'':>8s} {m.sum():6d} {ll(pm, y)[m].mean():8.4f}"
        for _, d in runs:
            lm = ll(d["p"], y)
            row += f" {lm[m].mean():10.4f} {(lm[m] - ll(pm, y)[m]).sum() / len(y):+8.4f}"
        print(row)
    row = f"{'ВСЕГО':>18s} {len(y):6d} {ll(pm, y).mean():8.4f}"
    for _, d in runs:
        lm = ll(d["p"], y)
        row += f" {lm.mean():10.4f} {(lm - ll(pm, y)).mean():+8.4f}"
    print(row)
    print(f"\nдля справки: подбрасывание монеты даёт {-np.log(0.5):.4f}")


if __name__ == "__main__":
    main()
