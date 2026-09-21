"""A large random search, done so that it cannot fool us.

The idea is Roman's: spend compute, try a great many configurations, re-check the
ones that stand out. That is sound as long as the re-check is on data the search
never saw, and docs/search_protocol.md says how — written before the first fit.
This script implements that file and nothing else, and stamps its SHA-256 into
everything it writes.

  python3 scripts/search.py --n 200                 # the screen, then the confirmation
  python3 scripts/search.py --smoke                 # one baseline, one candidate, own file
  python3 scripts/search.py --confirm-only          # confirm from the saved screen

The screen is resumable: every candidate is appended to results/search_screen.jsonl
the moment it is scored, and a rerun skips the ones already there.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(ROOT / "scripts" / "simulation" / "scripts"))
import lab  # noqa: E402
import market_eval as ME  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
# a smoke run writes beside the real screen, never into it
SCREEN = RESULTS / ("search_smoke.jsonl" if "--smoke" in sys.argv else "search_screen.jsonl")
PROTOCOL = ROOT / "docs" / "search_protocol.md"

SEED = 20260921
CUT_A = pd.Timestamp("2021-06-10")     # training for the search ends here
TOP_K = 5
FEATS = "everyz"

# the final configuration, as lab.fit spells it
BASELINE = {"params": {"num_leaves": 63, "learning_rate": 0.03, "min_data_in_leaf": 100,
                       "feature_fraction": 0.9, "bagging_fraction": 0.9, "lambda_l2": 5.0,
                       "extra_trees": True, "max_bin": 255, "path_smooth": 0.0},
            "halflife": 6.0, "weight": "none"}


def sample(rng: np.random.Generator) -> dict:
    """One candidate, every setting drawn independently — the table in the
    protocol, in the protocol's order."""
    return {"params": {
                "num_leaves": int(rng.choice([15, 31, 63, 127, 255])),
                "learning_rate": float(10 ** rng.uniform(np.log10(0.01), np.log10(0.1))),
                "min_data_in_leaf": int(rng.choice([20, 50, 100, 200, 400])),
                "feature_fraction": float(rng.uniform(0.5, 1.0)),
                "bagging_fraction": float(rng.uniform(0.5, 1.0)),
                "lambda_l2": float(10 ** rng.uniform(np.log10(0.1), np.log10(50))),
                "extra_trees": bool(rng.choice([True, False])),
                "max_bin": int(rng.choice([63, 255])),
                "path_smooth": float(rng.choice([0.0, 1.0, 10.0, 50.0]))},
            "halflife": float(rng.choice([3, 4, 6, 8, 12, 0])),
            "weight": str(rng.choice(["none", "prem1.5", "prem2", "prem3", "top2", "top4"]))}


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(d, n=4000, seed=42, level=95.0):
    rng = np.random.default_rng(seed)
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    a = (100.0 - level) / 2
    return [float(x) for x in np.percentile(b, [a, 100.0 - a])]


def protocol_hash() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def screen(B, cols, n: int, rowsA, premA) -> list[dict]:
    """Candidates 0..n-1 plus the baseline under five seeds, each scored on
    window A. Resumes from the jsonl."""
    done = {}
    if SCREEN.exists():
        for line in SCREEN.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    yA = B.y_all[rowsA]
    rng = np.random.default_rng(SEED)
    jobs = [(f"base-s{k}", BASELINE, 42 + 1000 * k) for k in range(5)]
    jobs += [(f"c{i:04d}", sample(rng), 42) for i in range(n)]
    if "--smoke" in sys.argv:
        jobs = jobs[:1] + jobs[5:6]          # one baseline, one candidate
    h = protocol_hash()
    for jid, cfg, seed0 in jobs:
        if jid in done:
            continue
        t0 = time.time()
        pr = lab.fit(B, cols, CUT_A, seeds=1, tta=True, seed0=seed0,
                     halflife=cfg["halflife"], weight=cfg["weight"],
                     params_over=cfg["params"])
        p = pr(rowsA)
        rec = {"id": jid, "seed0": seed0, "cfg": cfg,
               "ll_A_corpus": float(ll(p, yA).mean()),
               "ll_A_prem": float(ll(p[premA], yA[premA]).mean()),
               "trees": int(pr.n_trees), "seconds": round(time.time() - t0, 1),
               "protocol_sha256": h}
        with SCREEN.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[jid] = rec
        print(f"{jid}  prem {rec['ll_A_prem']:.4f}  corpus {rec['ll_A_corpus']:.4f}  "
              f"{rec['trees']}t  {rec['seconds']:.0f}s", flush=True)
    return list(done.values())


def confirm(B, cols, recs: list[dict]) -> dict:
    """The top K on window A and the baseline, refitted to the main cutoff in
    the full deployment stack and scored on window B and the priced test."""
    cands = sorted((r for r in recs if r["id"].startswith("c")),
                   key=lambda r: r["ll_A_prem"])[:TOP_K]
    base_band = [r["ll_A_prem"] for r in recs if r["id"].startswith("base-s")]
    base_A = float(np.mean(base_band))
    y_corp, y_q = B.y_all[B.post], B.jy[B.qte]
    pm = np.clip(B.p_mkt[B.qte], 1e-6, 1 - 1e-6)

    def full(cfg):
        pr = lab.fit(B, cols, B.cutoff, seeds=3, mirror_train=True, tta=True,
                     halflife=cfg["halflife"], weight=cfg["weight"],
                     params_over=cfg["params"])
        return pr(B.post), pr(B.jidx[B.qte])

    print("\nconfirmation: baseline in the full stack", flush=True)
    b_corp, b_q = full(BASELINE)
    out = {"protocol_sha256": protocol_hash(), "n_screened": sum(r["id"].startswith("c") for r in recs),
           "seed_band_A_prem": {"mean": base_A, "min": float(np.min(base_band)),
                                "max": float(np.max(base_band)), "values": base_band},
           "baseline_B": {"ll_corpus": float(ll(b_corp, y_corp).mean()),
                          "ll_prem": float(ll(b_corp[B.prem], y_corp[B.prem]).mean()),
                          "ll_quoted": float(ll(b_q, y_q).mean()),
                          "ll_market": float(ll(pm, y_q).mean())},
           "candidates": []}
    for r in cands:
        print(f"confirmation: {r['id']}", flush=True)
        c_corp, c_q = full(r["cfg"])
        d_corp = ll(b_corp, y_corp) - ll(c_corp, y_corp)
        d_q = ll(b_q, y_q) - ll(c_q, y_q)
        g_q = ll(pm, y_q) - ll(c_q, y_q)
        rec = {"id": r["id"], "cfg": r["cfg"],
               "lead_A_prem": base_A - r["ll_A_prem"],
               "delta_B_corpus": {"delta": float(d_corp.mean()), "ci95": boot(d_corp)},
               "delta_B_confirm": {"delta": float(d_corp[B.conf].mean()),
                                   "ci95": boot(d_corp[B.conf])},
               "delta_B_prem": {"delta": float(d_corp[B.prem].mean()),
                                "ci95": boot(d_corp[B.prem]),
                                "ci99": boot(d_corp[B.prem], level=99.0)},
               "delta_quoted": {"delta": float(d_q.mean()), "ci95": boot(d_q)},
               "gap_to_close": {"gap": float(g_q.mean()), "ci95": boot(g_q)}}
        rec["survives"] = bool(rec["delta_B_prem"]["ci99"][0] > 0
                               and rec["delta_B_corpus"]["delta"] >= 0)
        out["candidates"].append(rec)
        print(f"  lead on A {rec['lead_A_prem']:+.4f} · on B premium "
              f"{rec['delta_B_prem']['delta']:+.4f} 99% {rec['delta_B_prem']['ci99']} · "
              f"corpus {rec['delta_B_corpus']['delta']:+.4f} · "
              f"{'SURVIVES' if rec['survives'] else 'does not survive'}", flush=True)
        (RESULTS / "search.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    n = int(ME.arg("--n", "200"))
    B = lab.Bench("l6")
    lab.BENCH = B
    cols = ME.resolve(FEATS)
    a = (B.dt > np.datetime64(CUT_A, "D")) & (B.dt <= np.datetime64(B.cutoff, "D")) & B.nd
    rowsA = np.where(a)[0]
    premA = B.prem_all[rowsA]
    print(f"protocol {protocol_hash()[:16]} · window A {len(rowsA):,} bouts, "
          f"premium {int(premA.sum()):,} · training to {CUT_A.date()}", flush=True)
    recs = screen(B, cols, n, rowsA, premA) if "--confirm-only" not in sys.argv else \
        [json.loads(x) for x in SCREEN.read_text().splitlines()]
    if "--smoke" in sys.argv:
        return
    confirm(B, cols, recs)


if __name__ == "__main__":
    main()
