"""Is the edge where the thesis says it is?

This project exists on one claim: that regional and club lines are softer than
world-level ones, because nobody sharp is betting a six-rounder in Tijuana. Every
number measured so far is an average over the quoted set, which cannot say
anything about that — an average hides exactly the thing the claim is about.

So: split the quoted test bouts by how big the fight was and look at the edge
inside each slice. If the thesis is right, the gap to the closing line should
NARROW as the level drops, and the closing-line value should GROW. If the model
does relatively better on world title fights than on eight-round club shows, the
thesis is not just unproven — it is backwards, and the whole plan of chasing the
regional tail should be dropped.

Level is read off things known before the bell and never off the price:
scheduled rounds first (a four is a club fight and a twelve is a title fight, and
that is what the distance is FOR), then the size of the card and how much of a
record the two men have.

CLV is the honest column. ROI on a few hundred bouts tells you nothing; CLV
converges about twenty times faster, and it is the one number here the model
earns without ever seeing a price.

  ./venv/bin/python scripts/regional.py v10-mirror
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
EDGE = 0.02          # fixed in advance, not chosen from the table


def ll(p, y):
    return -np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-9, 1))


def boot(d, n=3000, seed=42):
    if len(d) < 12:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return tuple(np.percentile(b, [2.5, 97.5]))


def fair(xa, xb):
    ia, ib = 1 / xa, 1 / xb
    s = ia + ib
    return ia / s, ib / s


def clv_roi(p, y, oa, ob, ca, cb):
    """Bet our disagreements at the OPEN, mark to the close. In probability,
    because in price a random side already collects the margin difference."""
    ok = np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1)
    ea, eb = p - 1 / np.where(ok, oa, np.inf), (1 - p) - 1 / np.where(ok, ob, np.inf)
    ba = (ea > EDGE) & (ea >= eb) & ok
    bb = (eb > EDGE) & (eb > ea) & ok
    sel = ba | bb
    if sel.sum() < 12:
        return 0, np.nan, (np.nan, np.nan), np.nan
    fo_a, fo_b = fair(oa[sel], ob[sel])
    fc_a, fc_b = fair(ca[sel], cb[sel])
    moved = np.where(ba[sel], fc_a - fo_a, fc_b - fo_b)
    took = np.where(ba[sel], oa[sel], ob[sel])
    won = np.where(ba[sel], y[sel] == 1, y[sel] == 0)
    pnl = np.where(won, took - 1.0, -1.0)
    return int(sel.sum()), float(moved.mean()), boot(moved), float(pnl.mean())


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "v10-mirror"
    d = np.load(CACHE / "preds" / f"{label}.npz", allow_pickle=True)
    p, y, pm = d["p"], d["y"], d["p_mkt"]
    ca, cb, oa, ob, key = d["ca"], d["cb"], d["oa"], d["ob"], d["key"]
    f = pd.read_parquet(CACHE / f"feats_card_v{F.FEATS_VERSION}.parquet",
                        columns=["sched_rounds", "card_size", "ntrue_min",
                                 "title_lvl", "is_title"]).iloc[key].reset_index(drop=True)
    sched = f["sched_rounds"].to_numpy(float)

    print(f"{label}: {len(y):,} quoted test bouts · "
          f"model {ll(p, y).mean():.4f} · market {ll(pm, y).mean():.4f}\n")
    print("Порог ставки зафиксирован заранее: 2% преимущества по цене открытия.\n")

    bands = [("4-6 раундов (клубный)", (sched >= 4) & (sched <= 6)),
             ("8 раундов (регионал)", sched == 8),
             ("10 раундов (нац./конт.)", sched == 10),
             ("12 раундов (титульный)", sched == 12),
             ("дистанция неизвестна", ~np.isfinite(sched))]
    hdr = (f"{'уровень':26s} {'n':>5s} {'модель':>8s} {'рынок':>8s} {'разрыв':>8s} "
           f"{'ставок':>7s} {'CLV':>8s} {'95% интервал':>18s} {'ROI':>8s}")
    print(hdr); print("-" * len(hdr))
    for name, m in bands:
        if m.sum() < 25:
            continue
        n, c, (lo, hi), roi = clv_roi(p[m], y[m], oa[m], ob[m], ca[m], cb[m])
        ci = f"[{lo:+.4f},{hi:+.4f}]" if np.isfinite(lo) else "—"
        print(f"{name:26s} {m.sum():5d} {ll(p[m], y[m]).mean():8.4f} "
              f"{ll(pm[m], y[m]).mean():8.4f} "
              f"{ll(p[m], y[m]).mean() - ll(pm[m], y[m]).mean():+8.4f} "
              f"{n:7d} {c:+8.4f} {ci:>18s} {roi:+7.1%}")

    # the same cut on how much record the two men carry, which separates a
    # padded prospect's card from a world-level one better than the distance does
    nt = f["ntrue_min"].to_numpy(float)
    print()
    bands2 = [("менее 8 боёв у слабейшего", nt < 8),
              ("8-15", (nt >= 8) & (nt < 15)),
              ("15-25", (nt >= 15) & (nt < 25)),
              ("25+", nt >= 25)]
    print(hdr); print("-" * len(hdr))
    for name, m in bands2:
        m = m & np.isfinite(nt)
        if m.sum() < 25:
            continue
        n, c, (lo, hi), roi = clv_roi(p[m], y[m], oa[m], ob[m], ca[m], cb[m])
        ci = f"[{lo:+.4f},{hi:+.4f}]" if np.isfinite(lo) else "—"
        print(f"{name:26s} {m.sum():5d} {ll(p[m], y[m]).mean():8.4f} "
              f"{ll(pm[m], y[m]).mean():8.4f} "
              f"{ll(p[m], y[m]).mean() - ll(pm[m], y[m]).mean():+8.4f} "
              f"{n:7d} {c:+8.4f} {ci:>18s} {roi:+7.1%}")

    # a third proxy, independent of both: was a belt on the line, and how big
    # was the card. A finding this consequential should agree from more than
    # two angles or not be reported.
    tl = f["title_lvl"].to_numpy(float)
    cs = f["card_size"].to_numpy(float)
    print()
    bands3 = [("без пояса, карточка <8", (np.nan_to_num(tl) == 0) & (cs < 8)),
              ("без пояса, карточка 8+", (np.nan_to_num(tl) == 0) & (cs >= 8)),
              ("регион./нац. пояс", (tl >= 1) & (tl <= 2)),
              ("конт./межд./мировой", tl >= 3)]
    print(hdr); print("-" * len(hdr))
    for name, m in bands3:
        m = np.asarray(m) & np.isfinite(cs)
        if m.sum() < 25:
            continue
        n, c, (lo, hi), roi = clv_roi(p[m], y[m], oa[m], ob[m], ca[m], cb[m])
        ci = f"[{lo:+.4f},{hi:+.4f}]" if np.isfinite(lo) else "—"
        print(f"{name:26s} {m.sum():5d} {ll(p[m], y[m]).mean():8.4f} "
              f"{ll(pm[m], y[m]).mean():8.4f} "
              f"{ll(p[m], y[m]).mean() - ll(pm[m], y[m]).mean():+8.4f} "
              f"{n:7d} {c:+8.4f} {ci:>18s} {roi:+7.1%}")

    print("\nЧитать так: тезис верен, если при движении ВНИЗ по уровню разрыв к "
          "закрытию\nсжимается, а CLV растёт. Если наоборот — тезис перевёрнут.")


if __name__ == "__main__":
    main()
