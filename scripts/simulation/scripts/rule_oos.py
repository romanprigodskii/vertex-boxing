"""Does the level rule survive an era it was never chosen on?

`regional.py` found that closing-line value quadruples from club fights to title
fights and picked a filter off the back of it — 12 rounds or a continental/world
belt — which then returned +11.3% ROI. Those cut-points were chosen after looking
at the table, on the window 2023-06 to 2026-07. Anything measured again on that
window, including new bouts added to it, is measured on the era that produced the
rule.

The honest test is a different era. The corpus holds 16,142 title fights before
the model's cutoff and 2,251 of them already carry odds, so the data exists —
what is missing is a model whose predictions there are out of sample. This trains
one with an earlier cutoff and applies the SAME rule, unchanged, to the window
between the two cutoffs.

Nothing here is fitted. The threshold is 2%, the filter is 12 rounds or a
continental/world belt, both copied from regional.py without adjustment. If the
high-level slice does not beat the low-level slice on a window the rule never
saw, the rule is an artefact of looking at a table.

  ./venv/bin/python scripts/rule_oos.py                 # cutoff 2021-06-10
  ./venv/bin/python scripts/rule_oos.py --cutoff 2020-06-10 --seeds 5
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(ROOT / "scripts" / "simulation" / "scripts"))
import lab  # noqa: E402
import market_eval as ME  # noqa: E402
from src import features as F  # noqa: E402

EDGE = 0.02          # copied from regional.py, not re-chosen


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
    ok = np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1)
    ea = p - 1 / np.where(ok, oa, np.inf)
    eb = (1 - p) - 1 / np.where(ok, ob, np.inf)
    ba = (ea > EDGE) & (ea >= eb) & ok
    bb = (eb > EDGE) & (eb > ea) & ok
    sel = ba | bb
    if sel.sum() < 12:
        return 0, np.nan, (np.nan, np.nan), np.nan, (np.nan, np.nan)
    fo_a, fo_b = fair(oa[sel], ob[sel])
    fc_a, fc_b = fair(ca[sel], cb[sel])
    moved = np.where(ba[sel], fc_a - fo_a, fc_b - fo_b)
    took = np.where(ba[sel], oa[sel], ob[sel])
    won = np.where(ba[sel], y[sel] == 1, y[sel] == 0)
    pnl = np.where(won, took - 1.0, -1.0)
    return (int(sel.sum()), float(moved.mean()), boot(moved),
            float(pnl.mean()), boot(pnl))


def main() -> None:
    cut = pd.Timestamp(ME.arg("--cutoff", "2021-06-10"))
    seeds = int(ME.arg("--seeds", "3"))
    B = lab.Bench("card")
    print(f"\nOUT-OF-SAMPLE ТЕСТ ПРАВИЛА · обучение до {cut.date()}, "
          f"проверка на {cut.date()} → {B.cutoff.date()}", flush=True)
    print("Правило и порог скопированы из regional.py без изменений.\n", flush=True)

    cols = ME.resolve("everyx")
    pr = lab.fit(B, cols, cut, seeds=seeds, tta=True)

    # quoted bouts strictly between the two cutoffs: unseen by this model, and
    # from an era the rule was never looked at
    qdt = B.df["dt"].to_numpy()[B.jidx]
    win = (qdt > np.datetime64(cut, "D")) & (qdt <= np.datetime64(B.cutoff, "D"))
    rows = B.jidx[win]
    if win.sum() < 100:
        print(f"only {win.sum()} quoted bouts in the window — pick an earlier cutoff")
        return
    p = pr(rows)
    y = B.jy[win]
    ca, cb = B.ca[win], B.cb[win]
    oa, ob = B.oa[win], B.ob[win]

    f = pd.read_parquet(ROOT / "imports" / "staging" /
                        f"feats_card_v{F.FEATS_VERSION}.parquet",
                        columns=["sched_rounds", "title_lvl"]).iloc[rows].reset_index(drop=True)
    sch = np.nan_to_num(f["sched_rounds"].to_numpy(float), nan=0)
    tl = np.nan_to_num(f["title_lvl"].to_numpy(float), nan=0)
    top = (sch >= 12) | (tl >= 3)
    low = (sch <= 8) & (tl < 1)

    print(f"котируемых боёв в окне: {win.sum():,} · из них верхних {top.sum():,} · "
          f"нижних {low.sum():,}")
    print(f"модель {ll(p, y).mean():.4f} · рынок "
          f"{ll(B.p_mkt[win], y).mean():.4f}\n")
    hdr = (f"{'срез':34s} {'ставок':>7s} {'CLV':>9s} {'95%':>20s} "
           f"{'ROI':>8s} {'95%':>18s}")
    print(hdr); print("-" * len(hdr))
    for name, m in (("ВСЕ котируемые", np.ones(len(y), bool)),
                    ("низкий уровень (<=8р, без пояса)", low),
                    ("ВЕРХНИЙ (12р или конт./мир. пояс)", top)):
        n, c, (lo, hi), roi, (rlo, rhi) = clv_roi(p[m], y[m], oa[m], ob[m],
                                                  ca[m], cb[m])
        if not n:
            continue
        print(f"{name:34s} {n:7d} {c:+9.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{roi:+7.1%} [{rlo:+.1%},{rhi:+.1%}]")
    print("\nПравило подтверждается, только если ВЕРХНИЙ срез даёт CLV заметно "
          "выше нижнего\nна окне, которого правило не видело.")


if __name__ == "__main__":
    main()
