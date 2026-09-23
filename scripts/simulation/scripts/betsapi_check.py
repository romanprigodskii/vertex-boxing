"""The scoreboard on a third price feed: BetsAPI, read against the same model.

ProBoxingOdds is the feed every number in the report is scored against, and the
re-crawl of it (robustness_pbo_v3.json) is the same site parsed again. BetsAPI is
a different source altogether: Bet365's own line, stamped, with the price at the
moment the bout went in-play as the close. It starts at the end of 2020.

The run it reads is market_eval.py with `--cutoff 2023-06-10`, the published
cutoff, so the model is the published model — this script checks that, bout by
bout — and on the bouts both feeds price the only thing that differs is the
price. Three questions:

  1. Do the two feeds agree about the bouts they share? Same favourite, how far
     apart the de-vigged prices sit, which one forecasts better.
  2. Does the gap hold on the shared bouts, and on the bouts only BetsAPI prices?
  3. Does Bet365 price the club level the thesis was aimed at, and if it does,
     what does the model do there?

Plus one check of the feed itself: BetsAPI's own result field against the
corpus, which says whether its first-named fighter is the one its first price
belongs to.

  python3 scripts/betsapi_check.py final-close final-close-betsapi --json results/diagnostics/betsapi.json
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
FEED = CACHE / "odds_external" / "betsapi.parquet"


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(x, n=3000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


def block(p, m, y) -> dict:
    g = ll(m, y) - ll(p, y)
    return {"n": int(len(y)), "model": float(ll(p, y).mean()),
            "market": float(ll(m, y).mean()), "gap": float(g.mean()),
            "ci": boot(g) if len(y) >= 30 else None}


def result_check(tag: str) -> dict:
    """BetsAPI's `ss` against the corpus's winner, joined by names alone.

    Independent of market_eval's join on purpose: this asks whether the FEED
    names its fighters in the order of its prices, and a check that went through
    the join it is checking would be circular."""
    od = pd.read_parquet(FEED)
    od = od[od["result"].isin(["a", "b"])].copy()
    od["dt"] = pd.to_datetime(od["date"])
    od["na"], od["nb"] = od["a"].map(ME.norm), od["b"].map(ME.norm)
    sym = pd.read_parquet(CACHE / f"sym_{tag}.parquet",
                          columns=["dt", "a", "a_name", "b_name", "winner_id", "is_draw"])
    sym = sym[~sym["is_draw"]]
    sym["na"], sym["nb"] = sym["a_name"].map(ME.norm), sym["b_name"].map(ME.norm)
    sym["won"] = np.where(sym["winner_id"].astype(str) == sym["a"].astype(str),
                          sym["na"], sym["nb"])
    m = od.merge(sym[["dt", "na", "nb", "won"]], on=["na", "nb"], suffixes=("", "_c"))
    m = m[(m["dt"] - m["dt_c"]).abs().dt.days <= 1].drop_duplicates("event_id")
    agree = np.where(m["result"] == "a", m["na"], m["nb"]) == m["won"]
    fav_won = np.where(m["close_a"] < m["close_b"], m["na"], m["nb"]) == m["won"]
    return {"n": int(len(m)), "result_agrees": float(agree.mean()),
            "favourite_won": float(fav_won.mean())}


def main() -> None:
    args, skip = [], False
    for x in sys.argv[1:]:
        if skip:
            skip = False
        elif x.startswith("--"):
            skip = x in ("--tag", "--json")
        else:
            args.append(x)
    base_label = args[0] if args else "final-close"
    bets_label = args[1] if len(args) > 1 else "final-close-betsapi"
    tag = ME.arg("--tag", "l6")
    a = np.load(CACHE / "preds" / f"{base_label}.npz")
    b = np.load(CACHE / "preds" / f"{bets_label}.npz")
    A = pd.DataFrame({"key": a["key"], "mA": a["p_mkt"], "pA": a["p"], "y": a["y"],
                      "caA": a["ca"], "cbA": a["cb"]})
    B = pd.DataFrame({"key": b["key"], "mB": b["p_mkt"], "pB": b["p"], "yB": b["y"],
                      "caB": b["ca"], "cbB": b["cb"], "blB": b["p_blend"]})
    both = A.merge(B, on="key")
    assert (both["y"] == both["yB"]).all(), "the two runs disagree on who won"
    # the same model or nothing: with the cutoff pinned, a bout both runs score
    # must get the same prediction to the last digit
    same_model = bool(np.allclose(both["pA"], both["pB"], rtol=0, atol=1e-12))
    out: dict = {"base": base_label, "betsapi": bets_label,
                 "same_model_on_shared_bouts": same_model,
                 "n_base": int(len(A)), "n_betsapi": int(len(B)),
                 "n_shared": int(len(both))}
    print(f"{len(A):,} test bouts on {base_label}, {len(B):,} on {bets_label}, "
          f"{len(both):,} shared · same model on the shared bouts: {same_model}")

    # 1. agreement on the shared bouts
    flip = (both["mA"] > 0.5) != (both["mB"] > 0.5)
    close_call = (both["mA"] - 0.5).abs().lt(0.05) | (both["mB"] - 0.5).abs().lt(0.05)
    out["agreement"] = {
        "favourite_differs": float(flip.mean()),
        "favourite_differs_outside_pickem": float(flip[~close_call].mean()),
        "median_abs_diff": float((both["mA"] - both["mB"]).abs().median()),
        "p90_abs_diff": float((both["mA"] - both["mB"]).abs().quantile(0.9)),
        "market_base": float(ll(both["mA"], both["y"]).mean()),
        "market_betsapi": float(ll(both["mB"], both["y"]).mean()),
        "betsapi_minus_base": boot(ll(both["mA"], both["y"]) - ll(both["mB"], both["y"])),
    }
    ag = out["agreement"]
    print(f"\n1. shared bouts: favourite differs on {ag['favourite_differs']:.1%} "
          f"({ag['favourite_differs_outside_pickem']:.1%} outside 45-55%), "
          f"median |Δp| {ag['median_abs_diff']:.3f}, 90th pct {ag['p90_abs_diff']:.3f}")
    print(f"   the market scores {ag['market_base']:.4f} on {base_label}'s price and "
          f"{ag['market_betsapi']:.4f} on BetsAPI's")

    # 2. the gap, on each part of the priced set
    onlyB = B[~B["key"].isin(A["key"])]
    out["gap"] = {
        "betsapi_all": block(B["pB"], B["mB"], B["yB"]),
        "shared_base_price": block(both["pA"], both["mA"], both["y"]),
        "shared_betsapi_price": block(both["pB"], both["mB"], both["y"]),
        "betsapi_only": block(onlyB["pB"], onlyB["mB"], onlyB["yB"]),
    }
    print("\n2. model − market")
    for k, v in out["gap"].items():
        ci = f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" if v["ci"] else ""
        print(f"   {k:22s} {v['n']:5,d}  model {v['model']:.4f}  market {v['market']:.4f}"
              f"  gap {v['gap']:+.4f} {ci}")

    # 3. the level: scheduled distance, as in REPORT 4.2
    f = pd.read_parquet(CACHE / F.cache_name("feats", tag), columns=["sched_rounds"])
    sr = f["sched_rounds"].to_numpy(float)
    def band(keys):
        r = np.nan_to_num(sr[np.asarray(keys)], nan=0)
        return np.select([r <= 6, r <= 8, r <= 10], ["≤6 or unknown", "8", "10"], "12")
    B["band"], A["band"] = band(B["key"]), band(A["key"])
    out["by_distance"] = {}
    print("\n3. by scheduled distance        base feed    BetsAPI   model − BetsAPI close")
    for bd in ["≤6 or unknown", "8", "10", "12"]:
        x = B[B["band"] == bd]
        rec = block(x["pB"], x["mB"], x["yB"])
        rec["n_base"] = int((A["band"] == bd).sum())
        rec["n_betsapi_only"] = int((~x["key"].isin(A["key"])).sum())
        out["by_distance"][bd] = rec
        ci = f"[{rec['ci'][0]:+.4f}, {rec['ci'][1]:+.4f}]" if rec["ci"] else ""
        print(f"   {bd:15s} {rec['n_base']:>12,d} {rec['n']:>10,d}   {rec['gap']:+.4f} {ci}"
              f"   ({rec['n_betsapi_only']:,} not on the base feed)")

    out["result_check"] = result_check(tag)
    rc = out["result_check"]
    print(f"\nBetsAPI's result against the corpus, by names alone: {rc['n']:,} bouts, "
          f"agree {rc['result_agrees']:.1%} · favourite won {rc['favourite_won']:.1%}")

    if "--json" in sys.argv:
        dst = Path(ME.arg("--json", ""))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
