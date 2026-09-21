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

THE 2026-08-02 RUN THAT FIRST REPORTED THIS WAS ON THE LEAKY MODEL — tag `card`,
the `everyx` of that day (judges' fields still in it), test-time averaging only.
Its ×1.8 and +4.2% are therefore not numbers about the model that ships. The
defaults are now the deployment configuration market_eval's headline runs: tag
`l6`, `everyz`, both orientations in training and at prediction, extremely
randomised trees. `--plain` drops the mirror and extra_trees, which is the
closest this script can come to the old configuration on the new features.

  python3 scripts/rule_oos.py --json results/rule_oos.json   # cutoff 2021-06-10
  python3 scripts/rule_oos.py --cutoff 2020-06-10 --seeds 5
"""

from __future__ import annotations

import json
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


def clv_roi(p, y, oa, ob, ca, cb, cap=np.inf, devig="power"):
    """CLV and ROI on the bets this model would strike, plus the return the
    closing price itself predicts for them.

    `cap` refuses any bet whose own price is at or above it. That is a rule
    about the price, not about the edge — knowable before the bell, and it
    exists because clv_money.py found the closing-line value is +0.045 in
    probability below 2.0 and MINUS 0.032 above 5.0. The average hid two
    opposite things.

    The third number, `imp`, is E[o*q]-1 with q the de-vigged closing
    probability: what these bets pay if the closing price is the truth. It has
    no outcomes in it, so its interval is roughly a seventh as wide as ROI's,
    and it is the one that says whether a positive ROI is edge or luck.
    """
    ok = np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1)
    ea = p - 1 / np.where(ok, oa, np.inf)
    eb = (1 - p) - 1 / np.where(ok, ob, np.inf)
    ba = (ea > EDGE) & (ea >= eb) & ok & (oa < cap)
    bb = (eb > EDGE) & (eb > ea) & ok & (ob < cap)
    sel = ba | bb
    nan2 = (np.nan, np.nan)
    if sel.sum() < 12:
        return 0, np.nan, nan2, np.nan, nan2, np.nan, nan2
    fo_a, fo_b = fair(oa[sel], ob[sel])
    fc_a, fc_b = fair(ca[sel], cb[sel])
    moved = np.where(ba[sel], fc_a - fo_a, fc_b - fo_b)
    took = np.where(ba[sel], oa[sel], ob[sel])
    won = np.where(ba[sel], y[sel] == 1, y[sel] == 0)
    pnl = np.where(won, took - 1.0, -1.0)
    q = ME.devig(1 / ca[sel], 1 / cb[sel], devig)
    imp = took * np.where(ba[sel], q, 1 - q) - 1.0
    return (int(sel.sum()), float(moved.mean()), boot(moved),
            float(pnl.mean()), boot(pnl), float(imp.mean()), boot(imp))


def main() -> None:
    cut = pd.Timestamp(ME.arg("--cutoff", "2021-06-10"))
    seeds = int(ME.arg("--seeds", "3"))
    tag = ME.arg("--tag", "l6")
    fset = ME.arg("--feats", "everyz")
    B = lab.Bench(tag)
    print(f"\nOUT-OF-SAMPLE ТЕСТ ПРАВИЛА · обучение до {cut.date()}, "
          f"проверка на {cut.date()} → {B.cutoff.date()}", flush=True)
    print("Правило и порог скопированы из regional.py без изменений.\n", flush=True)

    cols = ME.resolve(fset)
    stack = ({} if "--plain" in sys.argv
             else {"mirror_train": True, "params_over": {"extra_trees": True}})
    pr = lab.fit(B, cols, cut, seeds=seeds, tta=True, **stack)
    res: dict = {"cutoff_train": str(cut.date()), "window_end": str(B.cutoff.date()),
                 "tag": tag, "feats": fset, "seeds": seeds,
                 "config": "tta" if "--plain" in sys.argv else "tta+mirror+extra_trees",
                 "edge_threshold": EDGE, "slices": {}, "price_cap": {}}

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
    # keep what took the training to produce, so any further cut of this window
    # is a read of a file and not another quarter of an hour of boosting
    pdir = ROOT / "imports" / "staging" / "preds"
    pdir.mkdir(exist_ok=True)
    np.savez(pdir / f"rule-oos-{cut.date()}.npz", p=p, y=y, rows=rows, p_mkt=B.p_mkt[win],
             ca=ca, cb=cb, oa=oa, ob=ob)

    f = pd.read_parquet(ROOT / "imports" / "staging" / F.cache_name("feats", tag),
                        columns=["sched_rounds", "title_lvl"]).iloc[rows].reset_index(drop=True)
    sch = np.nan_to_num(f["sched_rounds"].to_numpy(float), nan=0)
    tl = np.nan_to_num(f["title_lvl"].to_numpy(float), nan=0)
    top = (sch >= 12) | (tl >= 3)
    low = (sch <= 8) & (tl < 1)

    print(f"котируемых боёв в окне: {win.sum():,} · из них верхних {top.sum():,} · "
          f"нижних {low.sum():,}")
    print(f"модель {ll(p, y).mean():.4f} · рынок "
          f"{ll(B.p_mkt[win], y).mean():.4f}\n")
    res |= {"n_quoted_window": int(win.sum()), "n_top": int(top.sum()),
            "n_low": int(low.sum()), "ll_model": float(ll(p, y).mean()),
            "ll_market": float(ll(B.p_mkt[win], y).mean()), "devig": B.dv}
    hdr = (f"{'срез':34s} {'ставок':>7s} {'CLV':>9s} {'95%':>20s} "
           f"{'ROI':>8s} {'95%':>18s} {'ROI по закр.':>13s}")
    print(hdr); print("-" * len(hdr))

    def line(name, m, cap=np.inf):
        r = clv_roi(p[m], y[m], oa[m], ob[m], ca[m], cb[m], cap=cap)
        n, c, (lo, hi), roi, (rlo, rhi), imp, (ilo, ihi) = r
        if not n:
            return None
        # the same bets priced at the close under the other reading of the
        # margin: clv_money.py found that which price band pays depends on it
        *_, imp_p, (plo, phi) = clv_roi(p[m], y[m], oa[m], ob[m], ca[m], cb[m], cap=cap,
                                        devig="proportional")
        print(f"{name:34s} {n:7d} {c:+9.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{roi:+7.1%} [{rlo:+.1%},{rhi:+.1%}] "
              f"{imp:+7.1%} [{ilo:+.1%},{ihi:+.1%}]  prop {imp_p:+.1%} [{plo:+.1%},{phi:+.1%}]")
        return {"bouts": int(m.sum()), "bets": n, "clv": c, "clv_ci": [lo, hi],
                "roi": roi, "roi_ci": [rlo, rhi],
                "roi_at_close": imp, "roi_at_close_ci": [ilo, ihi],
                "roi_at_close_proportional": imp_p,
                "roi_at_close_proportional_ci": [plo, phi]}

    for key, name, m in (("all", "ВСЕ котируемые", np.ones(len(y), bool)),
                         ("low", "низкий уровень (<=8р, без пояса)", low),
                         ("top", "ВЕРХНИЙ (12р или конт./мир. пояс)", top)):
        res["slices"][key] = line(name, m)
    t, lo_ = res["slices"]["top"], res["slices"]["low"]
    if t and lo_ and lo_["clv"]:
        res["clv_ratio_top_over_low"] = t["clv"] / lo_["clv"]
        print(f"\nверх / низ по CLV: ×{res['clv_ratio_top_over_low']:.2f}")

    # The second rule, and the reason this script now carries a price cap.
    # clv_money.py found the upper tier's closing-line value is entirely a
    # favourites effect: +0.045 in probability below 2.0, -0.032 above 5.0.
    # That was read off a table on the 2023-06+ window, so it is exactly the
    # kind of number that shrinks when tested elsewhere — which is what this
    # window is for. Cut-points are declared here and not tuned.
    print("\n=== потолок цены поверх того же фильтра по уровню ===")
    print("(правило о цене, не об эдже; выбрано по механизму favourite-longshot,")
    print(" проверяется здесь на окне, которого оно не видело)")
    print(hdr); print("-" * len(hdr))
    for cap in (2.0, 3.0, 5.0):
        res["price_cap"][f"top_below_{cap:.0f}"] = line(f"ВЕРХНИЙ, цена < {cap:.0f}",
                                                       top, cap=cap)

    print("\nЧитать так: правило по уровню подтверждается, только если ВЕРХНИЙ "
          "срез даёт CLV\nзаметно выше нижнего. Правило по цене — только если "
          "полоса фаворитов держит\nCLV и здесь. А колонка «ROI по закрытию» "
          "говорит, чего ждать от денег: она узкая\nи в ней нет исходов боёв, "
          "так что расхождение с фактическим ROI — это удача, не эдж.")
    if "--json" in sys.argv:
        dst = Path(ME.arg("--json", ""))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(res, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
