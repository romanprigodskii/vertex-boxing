"""CLV and ROI are the same bets in two units. This converts between them.

The scoreboard reports closing-line value in probability and return in percent,
and reads the first as the trustworthy version of the second. That reading skips
a step. A bet is struck at a price carrying margin, and the movement has to cover
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

THE DE-VIG IS THE SAME AT BOTH ENDS, and that is a correction. Until 2026-09-21
this script split the margin out of the OPEN proportionally and out of the CLOSE
by the power method. Power moves probability from the longshot to the favourite
relative to proportional, so the difference was positive on a favourite and
negative on a longshot before the line had moved at all — and it reproduced
almost the whole of the published "the CLV is all on favourites, the line runs
away from us on longshots". Measured on the same opening price with zero
movement, the method difference alone was +0.0232 below 2.0 and -0.0345 above
5.0. Every table below is now computed under ONE method per column, both methods
are printed, and neither is chosen after looking.

Read alongside `regional.py`, which reports the CLV (proportional at both ends),
and `rule_oos.py`, which tests the level rule on an era it never saw.

  python3 scripts/clv_money.py final-close --tag l6 --json results/clv_money.json
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
import market_eval as ME  # noqa: E402
from src import features as F  # noqa: E402

CACHE = ROOT / "imports" / "staging"
EDGE = 0.02                        # fixed in advance, copied from regional.py
METHODS = ("proportional", "power")


def boot(x, n=3000, seed=42):
    # the same resampling as regional.py, draw for draw: the two scripts report
    # the same statistics on the same bets, and an interval that differed in the
    # third decimal between them would be a discrepancy nobody could explain
    rng = np.random.default_rng(seed)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return np.percentile(b, [2.5, 97.5])


def report(name, r, imp) -> dict:
    lo, hi = boot(r)
    lo2, hi2 = boot(imp)
    lo3, hi3 = boot(r - imp)
    print(f"\n{name}: {len(r):,} bets")
    print(f"   realised ROI        {r.mean():+7.2%}  [{lo:+.2%}, {hi:+.2%}]"
          f"   width {hi - lo:5.1%}")
    print(f"   closing-implied ROI {imp.mean():+7.2%}  [{lo2:+.2%}, {hi2:+.2%}]"
          f"   width {hi2 - lo2:5.1%}")
    print(f"   difference          {np.mean(r - imp):+7.2%}  [{lo3:+.2%}, {hi3:+.2%}]")
    return {"bets": int(len(r)),
            "roi": float(r.mean()), "roi_ci": [float(lo), float(hi)],
            "roi_at_close": float(imp.mean()), "roi_at_close_ci": [float(lo2), float(hi2)],
            "difference": float(np.mean(r - imp)), "difference_ci": [float(lo3), float(hi3)]}


def main() -> None:  # noqa: PLR0915
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    label = args[0] if args else "final-close"
    tag = ME.arg("--tag", "l6")
    d = np.load(CACHE / "preds" / f"{label}.npz", allow_pickle=True)
    p, y = d["p"], d["y"]
    ca, cb, oa, ob, key = d["ca"], d["cb"], d["oa"], d["ob"], d["key"]
    f = pd.read_parquet(CACHE / F.cache_name("feats", tag),
                        columns=["sched_rounds", "title_lvl"]).iloc[key].reset_index(drop=True)
    dts = pd.read_parquet(CACHE / f"sym_{tag}.parquet",
                          columns=["dt"]).iloc[key]["dt"].reset_index(drop=True)
    out: dict = {"label": label, "tag": tag, "edge_threshold": EDGE,
                 "n_quoted_test": int(len(y))}

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
    sel = BA | BB
    m = sel & upper
    out["n_upper"] = int(upper.sum())

    print(f"{label}: {len(y):,} quoted test bouts · upper tier {int(upper.sum()):,}")

    # our side's fair probability at each end, under each method. The opening
    # board is de-vigged only where it exists — outside `ok` no bet is struck.
    def side(xa, xb, dv):
        q = np.full(len(xa), np.nan)
        k = np.isfinite(xa) & np.isfinite(xb) & (xa > 1) & (xb > 1)
        q[k] = ME.devig(1 / xa[k], 1 / xb[k], dv)
        return np.where(BA, q, 1 - q)

    fo = {dv: side(oa, ob, dv) for dv in METHODS}
    qc = {dv: side(ca, cb, dv) for dv in METHODS}

    out["by_method"] = {}
    for dv in METHODS:
        print(f"\n{'=' * 78}\n[{dv} de-vig at the open AND at the close]")
        blk: dict = {}
        for nm, k in [("all quoted", sel), ("upper tier", m)]:
            blk[nm.replace(" ", "_")] = report(
                f"  {nm}", np.where(won[k], took[k] - 1, -1.0), took[k] * qc[dv][k] - 1)
        print(f"\n   the arithmetic, per bet, in probability ({int(m.sum()):,} bets)")
        pay, fo_m, qc_m = np.mean(1 / took[m]), fo[dv][m].mean(), qc[dv][m].mean()
        keeps = float(np.mean(1 / took[m] - fo[dv][m]))
        clv = qc[dv][m] - fo[dv][m]
        clo, chi = boot(clv)
        net = float(np.mean(qc[dv][m] - 1 / took[m]))
        print(f"     we pay          {pay:.4f}")
        print(f"     open fair       {fo_m:.4f}  → the book keeps {keeps:+.4f}")
        print(f"     close fair      {qc_m:.4f}  → it came to us    "
              f"{clv.mean():+.4f} [{clo:+.4f}, {chi:+.4f}]  (this is the CLV)")
        print(f"     net                             {net:+.4f}")
        blk["upper_arithmetic"] = {"bets": int(m.sum()), "we_pay": float(pay),
                                   "open_fair": float(fo_m), "book_keeps": keeps,
                                   "close_fair": float(qc_m), "clv": float(clv.mean()),
                                   "clv_ci": [float(clo), float(chi)], "net": net}
        out["by_method"][dv] = blk

    # The sign is set by an average that hides its own shape. Both methods,
    # side by side: a pattern that holds under one and not the other is a
    # property of the method, not of the market.
    print(f"\n{'=' * 78}\nby the price taken, upper tier — each method at both ends")
    band = pd.cut(took[m], [1, 2, 3, 5, 1000], labels=["<2", "2-3", "3-5", "5+"])
    r_all = np.where(won[m], took[m] - 1, -1.0)
    rows = []
    for b in ["<2", "2-3", "3-5", "5+"]:
        k = np.asarray(band == b)
        if k.sum() < 10:
            continue
        row = {"price": b, "bets": int(k.sum()),
               "realised ROI": float(r_all[k].mean()),
               "share of spread": float(np.var(r_all[k]) * k.sum()
                                        / (np.var(r_all) * len(r_all)))}
        for dv in METHODS:
            s = dv[:4]
            row[f"keeps ({s})"] = float(np.mean(1 / took[m][k] - fo[dv][m][k]))
            row[f"CLV ({s})"] = float(np.mean(qc[dv][m][k] - fo[dv][m][k]))
            row[f"ROI at close ({s})"] = float(np.mean(took[m][k] * qc[dv][m][k] - 1))
        rows.append(row)
    tbl = pd.DataFrame(rows)
    show = tbl.copy()
    for c in show.columns:
        if c.startswith(("keeps", "CLV")):
            show[c] = show[c].map(lambda v: f"{v:+.4f}")
        elif "ROI" in c:
            show[c] = show[c].map(lambda v: f"{v:+.1%}")
        elif c == "share of spread":
            show[c] = show[c].map(lambda v: f"{v:.0%}")
    print(show.to_string(index=False))
    out["upper_by_price"] = rows

    # The quoted test in halves by date. This is NOT an out-of-sample test and
    # must not be read as one: regional.py chose the rule's cut-points off a
    # table built on all quoted bouts, so both halves are inside the window that
    # produced it. rule_oos.py is the out-of-sample test. What this shows is
    # narrower and still worth knowing — whether the realised return is a
    # property of the strategy or of one stretch of the calendar.
    print(f"\n{'=' * 78}\nthe quoted test in halves by date (not an out-of-sample test)")
    mid = dts.quantile(0.5)
    print(f"split at {pd.Timestamp(mid).date()} · closing-implied ROI under power")
    out["halves"] = {"split": str(pd.Timestamp(mid).date())}
    for nm, half in [("early half", (dts <= mid).to_numpy()),
                     ("late half ", (dts > mid).to_numpy())]:
        k = m & half
        if k.sum() < 30:
            continue
        out["halves"][nm.strip().replace(" ", "_")] = report(
            f"  {nm}", np.where(won[k], took[k] - 1, -1.0), took[k] * qc["power"][k] - 1)

    # If the sign is set by the margin at the open, the useful question is not
    # "is there an edge" but "how thin does the open have to be". Hold the
    # market's fair view fixed and re-price the same bet at a tighter board.
    # Nothing about the model or the selection moves.
    #
    # The margin has to go back on the way the method says it came off. The
    # first version re-priced with o' = 1/(f_o·w) under both methods, which is
    # the proportional structure; under power that silently moved margin from
    # the longshots onto the favourites as well as shrinking it, and the
    # "actual" row came out at -1.3% where the same bets at their real prices
    # return -6.1%. Now the actual row uses each bout's own overround and
    # reproduces the real prices exactly under either method — that equality is
    # the check that the re-pricing is the inverse of the de-vig.
    def remargin(f, w, dv):
        if dv == "proportional":
            return 1 / (f * w)
        from scipy.optimize import brentq
        o2 = np.empty(len(f))
        ws = np.broadcast_to(w, f.shape)
        for i, (x, wi) in enumerate(zip(f, ws)):
            # implied = fair**c, both sides together summing to the overround;
            # c < 1 on an ordinary board, c > 1 on the few opening boards that
            # sum to under 100% — the same bracket devig() searches for 1/c
            c = brentq(lambda c, x=x, wi=wi: x ** c + (1 - x) ** c - wi, 1e-3, 400.0)
            o2[i] = x ** -c
        return o2

    print(f"\n{'=' * 78}\nwhat the same bets pay at a tighter opening board")
    ov = (1 / oa + 1 / ob)[m]
    print(f"the board we actually have: overround {ov.mean():.4f} "
          f"(median {np.median(ov):.4f})")
    fav = took[m] < 2.0
    out["tighter_board"] = {"overround_mean": float(ov.mean()),
                            "overround_median": float(np.median(ov)), "rows": []}
    for dv in METHODS:
        print(f"\n  [{dv} at both ends]")
        print(f"  {'overround':>10} {'we pay':>8} {'net (prob)':>11} "
              f"{'ROI by the close':>17}   {'favourites only':>16}")
        for w in (None, 1.05, 1.04, 1.03, 1.02):
            o2 = remargin(fo[dv][m], ov if w is None else w, dv)
            if w is None:
                assert np.allclose(o2, took[m], rtol=1e-6), "re-pricing is not the inverse"
            imp = o2 * qc[dv][m] - 1
            tag_ = "actual" if w is None else f"{w:.3f}"
            print(f"  {tag_:>10} {np.mean(1 / o2):>8.4f} "
                  f"{np.mean(qc[dv][m] - 1 / o2):>+11.4f} "
                  f"{imp.mean():>+17.2%}   {imp[fav].mean():>+16.2%}")
            out["tighter_board"]["rows"].append(
                {"method": dv, "overround": tag_, "roi_at_close": float(imp.mean()),
                 "roi_at_close_favourites": float(imp[fav].mean())})
    print("Selection is held fixed at the real board — a tighter open would also")
    print("change which bouts clear the 2% threshold, so read these as the value")
    print("of the SAME bets at a better price, not as a re-run of the strategy.")

    # ca/cb in the npz are close_* — the WORST price across the ten books, which
    # is the most generous reading for us. The saved run does not carry best_*,
    # so fetch it back through the same join on the same corpus rows: a claim
    # that the sign survives every reading of the board should not depend on a
    # different prediction file.
    print(f"\n{'=' * 78}\nthe same bets, with the closing board read three ways")
    jj = ME.join_odds(ME.build(tag)[0], verbose=False).set_index("index")
    have = [i for i in key if i in jj.index]
    out["closing_board_readings"] = []
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
            k = sel & upper & np.isfinite(xa) & np.isfinite(xb) & (xa > 1) & (xb > 1)
            line = f"  {nm:<22} {int(k.sum()):>4} bets  overround " \
                   f"{np.mean((1 / xa + 1 / xb)[k]):.4f} → "
            rec = {"reading": nm, "bets": int(k.sum()),
                   "overround": float(np.mean((1 / xa + 1 / xb)[k]))}
            for dv in METHODS:
                q2 = side(xa, xb, dv)
                v = float(np.mean(took[k] * q2[k] - 1))
                rec[f"roi_at_close_{dv}"] = v
                line += f"{dv[:4]} {v:+.2%}  "
            out["closing_board_readings"].append(rec)
            print(line)

    if "--json" in sys.argv:
        dst = Path(ME.arg("--json", ""))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
