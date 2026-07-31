"""Flat-stake betting into the closing price, vig included.

Log-loss is scored on the devigged probability, which is a fair way to compare
two forecasts and a dishonest way to talk about money: the price you actually
get carries the bookmaker's margin. This settles every test bout at the real
decimal number and reports ROI with a bootstrap interval.

  ./venv/bin/python scripts/bet_sim.py p-final2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PRED = Path(__file__).resolve().parents[3] / "imports" / "staging" / "preds"


def roi(p, y, ca, cb, edge: float, band: tuple[float, float] | None = None,
        pm=None) -> tuple[int, float, tuple[float, float]]:
    """Back A when the forecast beats A's implied price by `edge`, else B."""
    ia, ib = 1 / ca, 1 / cb                      # implied, WITH margin
    bet_a = p - ia > edge
    bet_b = (1 - p) - ib > edge
    sel = bet_a | bet_b
    if band is not None and pm is not None:
        sel &= (pm > band[0]) & (pm < band[1])
    if sel.sum() == 0:
        return 0, 0.0, (0.0, 0.0)
    won = np.where(bet_a[sel], y[sel] == 1, y[sel] == 0)
    price = np.where(bet_a[sel], ca[sel], cb[sel])
    pnl = np.where(won, price - 1.0, -1.0)
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(pnl, len(pnl), replace=True).mean() for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return int(sel.sum()), float(pnl.mean()), (float(lo), float(hi))


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "p-final2"
    d = np.load(PRED / f"{label}.npz", allow_pickle=True)
    y, ca, cb, pm = d["y"], d["ca"], d["cb"], d["p_mkt"]
    vig = (1 / ca + 1 / cb - 1).mean()
    print(f"{label}: {len(y):,} bouts · средняя маржа букмекера {vig:.2%}")
    for name in ("p", "p_blend"):
        if name not in d:
            continue
        p = d[name]
        print(f"\n  {'модель' if name == 'p' else 'бленд'}:")
        for edge in (0.0, 0.02, 0.05, 0.10):
            n, r, (lo, hi) = roi(p, y, ca, cb, edge)
            print(f"    порог {edge:.0%}: {n:5,} ставок · ROI {r:+7.2%} "
                  f"[95% {lo:+.2%}, {hi:+.2%}]")
        n, r, (lo, hi) = roi(p, y, ca, cb, 0.02, (0.3, 0.7), pm)
        print(f"    конкурентные 30-70%, порог 2%: {n:,} ставок · ROI {r:+.2%} "
              f"[95% {lo:+.2%}, {hi:+.2%}]")


if __name__ == "__main__":
    main()
