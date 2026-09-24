"""Pre-registration 3: which bouts to bet at the opening price, chosen on the
windows already used and tested once on a window that was not.

  python3 scripts/ev_rules.py dev --json results/rules_dev.json
  python3 scripts/ev_rules.py holdout --null 1000 --json results/rules_holdout_null.json
  python3 scripts/ev_rules.py holdout --real --json results/rules_holdout.json

THE CANDIDATES, fixed in docs/evalue_protocol_3.md before any was scored:

  R0  every bout (the rule of pre-registration 2)
  R1  no club bouts: scheduled for 8 rounds or more
  R2  title bouts: any belt on the line, or scheduled for 12
  R3  the report's upper tier: 12 rounds, or a continental, international or
      world belt

Every rule bets the same way. q is the opening price for corner A with the margin
removed by the power method. The alternative is the blend σ(λ·logit p +
(1-λ)·logit q), mixed uniformly over λ from 0.05 to 0.50. The stake is Kelly at
the posted opening decimals. Bouts where either fighter boxed in the 60 days
before are left out. A rule only chooses which bouts are bet.

THE CHOICE. `dev` scores every rule at real prices on the three windows already
used for this question. Each window is scored with its own model, trained before
it, and the log e-values are summed. The rule with the largest sum is written to
rules_dev.json. `holdout` reads that file, so the rule it tests is chosen by the
file and not by whoever runs it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(ROOT / "scripts" / "simulation" / "scripts"))
CACHE = ROOT / "imports" / "staging"
os.environ.setdefault("VERTEX_ODDS", str(CACHE / "odds_external" / "betsapi.parquet"))
import market_eval as ME  # noqa: E402
import ev_registry as REG  # noqa: E402
import ev_window as W  # noqa: E402

RULES = {
    "R0": ("every bout", lambda c: pd.Series(True, index=c.index)),
    "R1": ("scheduled for 8 rounds or more", lambda c: c["sched"] >= 8),
    "R2": ("any belt, or scheduled for 12", lambda c: (c["belt"] >= 1) | (c["sched"] == 12)),
    "R3": ("12 rounds, or a belt at rung 3+", lambda c: (c["sched"] >= 12) | (c["belt"] >= 3)),
}
DEV = [  # window, the model trained before it, the price file, (after, through]
    ("A 2016-06..2020-06", "win-a-model", "proboxingodds_v3.parquet", "2016-06-10", "2020-06-10"),
    ("B 2021-06..2023-06", "win-b-model", "odds_external/betsapi.parquet", "2021-06-10", "2023-06-10"),
    ("C 2023-06..2025-12", "openinfo-close", "odds_external/betsapi.parquet", "2023-06-10", "2025-12-31"),
]
HOLDOUT = ("H 2026-01..2026-07", "hold-model", "odds_external/betsapi.parquet", "2025-12-31", "2026-07-24")
ALPHA = 0.05


def bouts(label: str, odds: str, t0: str, t1: str, with_y: bool) -> pd.DataFrame:
    """The quiet bouts of a window with an opening price, the model's p, and the
    covariates the rules read. Outcomes only when asked for."""
    ME.ODDS = CACHE / odds
    npz = np.load(CACHE / "preds" / f"{label}.npz", allow_pickle=True)
    pc = pd.Series(npz["p_corp"], index=npz["key_corp"])
    df, _ = ME.build("l6")
    j = ME.join_odds(df, verbose=False)
    same = ~j["swap"].to_numpy(bool)
    d = pd.DataFrame({"key": j["index"].to_numpy(), "dt": j["dt"].to_numpy(),
                      "oa": np.where(same, j["open_a"], j["open_b"]).astype(float),
                      "ob": np.where(same, j["open_b"], j["open_a"]).astype(float)})
    d = d[(d["dt"] > pd.Timestamp(t0)) & (d["dt"] <= pd.Timestamp(t1)) & (d["oa"] > 1)
          & (d["ob"] > 1) & d["key"].isin(pc.index)].copy()
    d["p"] = pc.loc[d["key"]].to_numpy()
    d["q"] = ME.devig(1 / d["oa"].to_numpy(), 1 / d["ob"].to_numpy(), "power")
    sym = df[["dt", "a", "b"]]
    hist = pd.concat([sym[["dt", "a"]].rename(columns={"a": "f"}),
                      sym[["dt", "b"]].rename(columns={"b": "f"})])
    prev = {f: np.sort(g.values) for f, g in hist.groupby("f")["dt"]}

    def since(f, t):
        v = prev.get(f)
        v = v[v < t] if v is not None else []
        return (t - v[-1]) / np.timedelta64(1, "D") if len(v) else 1e9
    k = d["key"].to_numpy()
    d["gap"] = [min(since(a, t), since(b, t)) for a, b, t in
                zip(sym.loc[k, "a"].to_numpy(), sym.loc[k, "b"].to_numpy(), d["dt"].to_numpy())]
    d = d[d["gap"] >= W.QUIET_DAYS]
    cov = REG.covariates("l6")[["sched", "belt"]]
    d = d.join(cov, on="key")
    if with_y:
        yc = pd.Series(npz["y_corp"], index=npz["key_corp"])
        d["y"] = yc.loc[d["key"]].to_numpy().astype(int)
    return d.sort_values(["dt", "key"]).reset_index(drop=True)


def score(g: pd.DataFrame, y: np.ndarray) -> dict:
    p, q, da, db = (g[c].to_numpy() for c in ("p", "q", "oa", "ob"))
    e = W.e_real(p, q, da, db, y) if len(g) else 1.0
    return {"n": int(len(g)), "e_real": float(e), "log_e": float(np.log(max(e, 1e-300)))}


def main() -> None:
    stage = sys.argv[1]
    out: dict = {"stage": stage}
    if stage == "dev":
        out["windows"] = {}
        total = {r: 0.0 for r in RULES}
        for name, label, odds, t0, t1 in DEV:
            d = bouts(label, odds, t0, t1, with_y=True)
            out["windows"][name] = {}
            for r, (desc, rule) in RULES.items():
                g = d[rule(d).fillna(False).to_numpy(bool)]
                s = score(g, g["y"].to_numpy())
                out["windows"][name][r] = s
                total[r] += s["log_e"]
            print(name, {r: f"n {v['n']}, e {v['e_real']:.3g}" for r, v in out["windows"][name].items()})
        out["total_log_e"] = total
        out["selected"] = max(total, key=total.get)
        print("total log e:", {r: round(v, 2) for r, v in total.items()}, "→ selected", out["selected"])
    elif stage == "holdout":
        sel = json.loads((ROOT / "scripts" / "simulation" / "results" / "rules_dev.json").read_text())["selected"]
        name, label, odds, t0, t1 = HOLDOUT
        real = "--real" in sys.argv
        d = bouts(label, odds, t0, t1, with_y=real)
        out["selected"] = sel
        rule = RULES[sel][1]
        g = d[rule(d).fillna(False).to_numpy(bool)]
        if "--null" in sys.argv:
            reps = int(W.arg("--null", "1000"))
            rng = np.random.default_rng(20260924)
            E = np.array([score(g, (rng.random(len(g)) < g["q"].to_numpy()).astype(int))["e_real"]
                          for _ in range(reps)])
            out["null"] = {"reps": reps, "n": int(len(g)), "mean_e": float(E.mean()),
                           "se": float(E.std(ddof=1) / np.sqrt(reps)),
                           "share_ge_20": float((E >= 1 / ALPHA).mean()), "max": float(E.max())}
            print(json.dumps(out["null"], indent=1))
        elif real:
            out["primary"] = score(g, g["y"].to_numpy())
            out["primary"]["reject"] = out["primary"]["e_real"] >= 1 / ALPHA
            out["others"] = {}
            for r, (desc, rr) in RULES.items():
                gg = d[rr(d).fillna(False).to_numpy(bool)]
                out["others"][r] = score(gg, gg["y"].to_numpy())
            p, q, da, db, y = (g[c].to_numpy() for c in ("p", "q", "oa", "ob", "y"))
            out["flat_by_lambda"] = {}
            for lam in W.LAMBDAS:
                pb = W.sig(lam * W.logit(p) + (1 - lam) * W.logit(q))
                ba, bb = pb * da > 1, (1 - pb) * db > 1
                k = ba | bb
                rr_ = np.where(np.where(ba, y == 1, y == 0), np.where(ba, da, db) - 1, -1.0)[k]
                ci = None
                if k.sum() >= 30:
                    bs = [np.random.default_rng(i).choice(rr_, len(rr_)).mean() for i in range(2000)]
                    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                out["flat_by_lambda"][f"{lam:.2f}"] = {"bets": int(k.sum()),
                                                     "roi": float(rr_.mean()) if k.any() else None, "ci": ci}
            pr = out["primary"]
            print(f"HOLDOUT {sel}: n {pr['n']}  real e {pr['e_real']:.4g}  → "
                  f"{'REJECT' if pr['reject'] else 'no'} at e ≥ {1 / ALPHA:.0f}")
            print("others:", {r: f"n {v['n']}, e {v['e_real']:.3g}" for r, v in out["others"].items()})
            for lam, v in out["flat_by_lambda"].items():
                roi = "—" if v["roi"] is None else f"{v['roi']:+.1%}"
                print(f"  λ {lam}: {v['bets']:4d} bets  ROI {roi}  {v['ci']}")
    if "--json" in sys.argv:
        dst = Path(W.arg("--json", ""))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
