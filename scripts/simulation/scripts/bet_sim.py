"""Flat-stake betting into the real price, vig included — and the CLV.

Log-loss is scored on the devigged probability, which is a fair way to compare
two forecasts and a dishonest way to talk about money: the price you actually
get carries the bookmaker's margin. This settles every test bout at the real
decimal number and reports ROI with a bootstrap interval.

It also answers the question the ROI cannot answer on 400 bets. Nobody can bet
the close — the close is what the number became after the money spoke. What a
forecast is worth is measured by taking the OPEN and seeing which way the line
then moved: closing-line value converges about twenty times faster than profit
does, and it is the only number here a bankroll would care about.

  ./venv/bin/python scripts/bet_sim.py p-final2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PRED = Path(__file__).resolve().parents[3] / "imports" / "staging" / "preds"


def _boot(x, n=4000, seed=42):
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return tuple(np.percentile(b, [2.5, 97.5]))


def _pick(p, ca, cb, edge, band=None, pm=None):
    """Which side, if any. A book whose overround is smaller than the edge
    threshold can make both corners look like value at once — that is not two
    bets, it is a broken price, so the larger edge wins and the caller gets one
    stake instead of a silent bet on corner A at a longshot number."""
    ea, eb = p - 1 / ca, (1 - p) - 1 / cb
    bet_a = (ea > edge) & (ea >= eb)
    bet_b = (eb > edge) & (eb > ea)
    sel = bet_a | bet_b
    if band is not None and pm is not None:
        sel &= (pm > band[0]) & (pm < band[1])
    return sel, bet_a


def roi(p, y, ca, cb, edge, band=None, pm=None):
    sel, bet_a = _pick(p, ca, cb, edge, band, pm)
    if sel.sum() == 0:
        return 0, 0.0, (0.0, 0.0)
    won = np.where(bet_a[sel], y[sel] == 1, y[sel] == 0)
    price = np.where(bet_a[sel], ca[sel], cb[sel])
    pnl = np.where(won, price - 1.0, -1.0)
    return int(sel.sum()), float(pnl.mean()), _boot(pnl)


def _fair(xa, xb):
    """Proportional de-vig — enough for CLV, where only the DIFFERENCE between
    two de-vigged numbers is used and the method mostly cancels."""
    ia, ib = 1 / xa, 1 / xb
    s = ia + ib
    return ia / s, ib / s


def clv(p, y, oa, ob, ca, cb, edge):
    """Bet at the open, mark to the close.

    Measured in PROBABILITY, not in price, and that is not a detail. In this
    feed the opening line carries 5.8% margin and the close 7.9%, so simply
    taking a side at random and marking it to the close pays +3.5% before any
    skill at all — and an underdog-leaning strategy collects +5.3% of it. Price
    CLV would credit the model with the bookmaker's margin schedule. Once both
    ends are de-vigged, a coin flip scores exactly zero by construction and
    what is left is the line moving towards our pick.
    """
    ok = np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1)
    sel, bet_a = _pick(p, np.where(ok, oa, np.inf), np.where(ok, ob, np.inf), edge)
    sel &= ok
    if sel.sum() == 0:
        return 0, 0.0, (0.0, 0.0), 0.0, (0.0, 0.0), 0.0
    fo_a, fo_b = _fair(oa[sel], ob[sel])
    fc_a, fc_b = _fair(ca[sel], cb[sel])
    moved = np.where(bet_a[sel], fc_a - fo_a, fc_b - fo_b)
    took = np.where(bet_a[sel], oa[sel], ob[sel])
    closed = np.where(bet_a[sel], ca[sel], cb[sel])
    raw = (took / closed - 1.0).mean()
    won = np.where(bet_a[sel], y[sel] == 1, y[sel] == 0)
    pnl = np.where(won, took - 1.0, -1.0)
    return (int(sel.sum()), float(moved.mean()), _boot(moved),
            float(pnl.mean()), _boot(pnl), float(raw))


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "p-final2"
    d = np.load(PRED / f"{label}.npz", allow_pickle=True)
    y, ca, cb, pm = d["y"], d["ca"], d["cb"], d["p_mkt"]
    oa = d["oa"] if "oa" in d else np.full_like(ca, np.nan)
    ob = d["ob"] if "ob" in d else np.full_like(cb, np.nan)
    has_open = bool(np.isfinite(oa).any())
    # ANY closing price, not just the one literally called "close": the best-of-
    # market line is a closing price too, and a blend fed with it beats the close
    # by construction. The first version of this guard tested == "close" and let
    # the best-price run through, which produced a spurious +18% ROI.
    blend_saw_close = str(d["price"]) != "open" if "price" in d else True
    print(f"{label}: {len(y):,} боёв · маржа на закрытии {(1 / ca + 1 / cb - 1).mean():.2%}"
          + (f" · на открытии {(1 / oa + 1 / ob - 1)[np.isfinite(oa)].mean():.2%}"
             if has_open else ""))
    null = 0.0
    if has_open:
        ok = np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1)
        null = float(np.concatenate([oa[ok] / ca[ok] - 1, ob[ok] / cb[ok] - 1]).mean())
        print(f"  нулевой уровень: случайная сторона по открытию даёт {null:+.2%} "
              f"в цене и ровно 0 в вероятности — это разница маржи, не мастерство")
    for name in ("p", "p_blend"):
        if name not in d:
            continue
        p, who = d[name], "модель" if name == "p" else "бленд"
        print(f"\n  {who} — ставки по закрытию:")
        for edge in (0.0, 0.02, 0.05, 0.10):
            n, r, (lo, hi) = roi(p, y, ca, cb, edge)
            print(f"    порог {edge:.0%}: {n:5,} ставок · ROI {r:+7.2%} "
                  f"[95% {lo:+.2%}, {hi:+.2%}]")
        n, r, (lo, hi) = roi(p, y, ca, cb, 0.02, (0.3, 0.7), pm)
        print(f"    конкурентные 30-70%, порог 2%: {n:,} ставок · ROI {r:+.2%} "
              f"[95% {lo:+.2%}, {hi:+.2%}]")
        if has_open and not (name == "p_blend" and blend_saw_close):
            print(f"  {who} — ставки по ОТКРЫТИЮ, переоценка к закрытию (CLV):")
            for edge in (0.0, 0.02, 0.05):
                n, e, (elo, ehi), r, (lo, hi), raw = clv(p, y, oa, ob, ca, cb, edge)
                print(f"    порог {edge:.0%}: {n:5,} ставок · CLV {e:+.4f} "
                      f"вероятности [95% {elo:+.4f}, {ehi:+.4f}] · "
                      f"ROI {r:+7.2%} [95% {lo:+.2%}, {hi:+.2%}] "
                      f"· в цене {raw:+.2%} (нуль {null:+.2%})")
        elif has_open:
            print(f"  {who}: CLV не считается — бленд использует ЗАКРЫТИЕ как вход, "
                   "так что «обогнать закрытие» он обязан по построению")


if __name__ == "__main__":
    main()
