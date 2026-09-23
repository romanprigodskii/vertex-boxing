"""Money at the opening price, on a window the model never trained on — the test
of docs/evalue_protocol_2.md, written and timestamped before it ran.

The question, the same in every window: a bettor holds a model that knows only
what was known when the line opened, blends it with the opening price, and bets
Kelly at the opening decimals the book actually posted. Does wealth grow by more
than luck allows?

  python3 scripts/ev_window.py --label win-a --odds proboxingodds_v3.parquet \\
      --from 2016-06-10 --to 2020-06-10 --null 1000        # validation, reads no outcome
  python3 scripts/ev_window.py --label win-a --odds proboxingodds_v3.parquet \\
      --from 2016-06-10 --to 2020-06-10 --real --json results/window_a.json

THE E-VALUE. q is the opening price for corner A, the margin removed by the power
method. The alternative is p' = σ(λ·logit p + (1-λ)·logit q), where p is the
model and λ runs over ten fixed weights, 0.05 to 0.50, mixed uniformly. The stake
is Kelly at the posted decimals: back A where p'·d_a > 1, back B where
(1-p')·d_b > 1, otherwise nothing. At posted prices every bet has expectation at
most 1 for any true probability inside the band the two decimals leave. So the
product of the increments is a supermartingale under that null, with nothing
de-vigged in it, and its terminal value is an e-value. The mixture of such
products is one too.

`--real` is required to read outcomes. The null mode draws them from q instead.
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
CACHE = ROOT / "imports" / "staging"


def arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


ODDS = arg("--odds", "proboxingodds_v3.parquet")
os.environ["VERTEX_ODDS"] = str(Path(ODDS) if Path(ODDS).is_absolute() else CACHE / ODDS)
sys.path.insert(0, str(ROOT / "scripts" / "simulation" / "scripts"))
import market_eval as ME  # noqa: E402

LAMBDAS = np.round(np.arange(0.05, 0.501, 0.05), 2)
ALPHA = 0.05
QUIET_DAYS = 60       # neither man boxed in the 60 days before: no bout between open and bell
EPS = 1e-12


def sig(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def kelly_log(pa, da, db, y):
    f = np.clip((pa * da - 1) / np.maximum(da - 1, 1e-9), 0, 1)
    g = np.clip(((1 - pa) * db - 1) / np.maximum(db - 1, 1e-9), 0, 1)
    m = np.where(y == 1, 1 + f * (da - 1) - g, 1 - f + g * (db - 1))
    return np.log(np.maximum(m, EPS))


def lr_log(pa, q, y):
    pa, q = np.clip(pa, 1e-9, 1 - 1e-9), np.clip(q, 1e-9, 1 - 1e-9)
    return np.where(y == 1, np.log(pa) - np.log(q), np.log(1 - pa) - np.log(1 - q))


def mix_end(logs) -> float:
    s = np.array([x.sum() for x in logs])
    return float(np.exp(s.max() + np.log(np.mean(np.exp(s - s.max())))))


def e_real(p, q, da, db, y) -> float:
    return mix_end([kelly_log(sig(lam * logit(p) + (1 - lam) * logit(q)), da, db, y)
                    for lam in LAMBDAS])


def e_fair(p, q, y) -> float:
    return mix_end([lr_log(sig(lam * logit(p) + (1 - lam) * logit(q)), q, y) for lam in LAMBDAS])


def load(label: str, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """Every bout in (t0, t1] the price file gives an opening price for, with the
    model's p from the run `label` (which must have been trained to t0)."""
    npz = np.load(CACHE / "preds" / f"{label}.npz", allow_pickle=True)
    pc = pd.Series(npz["p_corp"], index=npz["key_corp"])
    df, _ = ME.build(ME.arg("--tag", "l6"))
    j = ME.join_odds(df, verbose=False)
    same = ~j["swap"].to_numpy(bool)
    d = pd.DataFrame({"key": j["index"].to_numpy(), "dt": j["dt"].to_numpy(),
                      "oa": np.where(same, j["open_a"], j["open_b"]).astype(float),
                      "ob": np.where(same, j["open_b"], j["open_a"]).astype(float)})
    d = d[(d["dt"] > t0) & (d["dt"] <= t1) & (d["oa"] > 1) & (d["ob"] > 1)
          & d["key"].isin(pc.index)].copy()
    d["p"] = pc.loc[d["key"]].to_numpy()
    d["q"] = ME.devig(1 / d["oa"].to_numpy(), 1 / d["ob"].to_numpy(), "power")
    # the quiet filter: a bout either man fought in the 60 days before this one
    # could fall between the opening price and the bell, and the model would
    # know its result while the price did not. Bet365's lines open a median 3
    # days and a 90th percentile 37 days before the bell (evalue_followup.json);
    # 60 covers that with room. `gap` is kept so 0/30/90 can be reported.
    sym = pd.read_parquet(CACHE / f"sym_{ME.arg('--tag', 'l6')}.parquet", columns=["dt", "a", "b"])
    hist = pd.concat([sym[["dt", "a"]].rename(columns={"a": "f"}),
                      sym[["dt", "b"]].rename(columns={"b": "f"})]).sort_values("dt")
    prev: dict = {}
    for f, t in zip(hist["f"].to_numpy(), hist["dt"].to_numpy()):
        prev.setdefault(f, []).append(t)
    for f in prev:
        prev[f] = np.array(prev[f])
    def since(f, t):
        v = prev.get(f)
        v = v[v < t] if v is not None else []
        return (t - v[-1]) / np.timedelta64(1, "D") if len(v) else 1e9
    fa, fb = sym.loc[d["key"], "a"].to_numpy(), sym.loc[d["key"], "b"].to_numpy()
    dts = d["dt"].to_numpy()
    d["gap"] = [min(since(a, t), since(b, t)) for a, b, t in zip(fa, fb, dts)]
    d["quiet"] = d["gap"] >= QUIET_DAYS
    if "--real" in sys.argv:
        yc = pd.Series(npz["y_corp"], index=npz["key_corp"])
        d["y"] = yc.loc[d["key"]].to_numpy().astype(int)
    return d.sort_values(["dt", "key"]).reset_index(drop=True)


def main() -> None:
    label = arg("--label", "")
    t0, t1 = pd.Timestamp(arg("--from", "")), pd.Timestamp(arg("--to", ""))
    d = load(label, t0, t1)
    q = d[d["quiet"]]
    print(f"[{label}] {ODDS}: {len(d):,} bouts with an opening price in ({t0.date()}, {t1.date()}], "
          f"{len(q):,} quiet (the primary set)")
    out: dict = {"label": label, "odds": ODDS, "from": str(t0.date()), "to": str(t1.date()),
                 "n_all": int(len(d)), "n_primary": int(len(q)), "threshold": 1 / ALPHA}

    if "--null" in sys.argv:
        rng = np.random.default_rng(20260923)
        reps = int(arg("--null", "1000"))
        p, qq, da, db = q["p"].to_numpy(), q["q"].to_numpy(), q["oa"].to_numpy(), q["ob"].to_numpy()
        E = np.array([e_real(p, qq, da, db, (rng.random(len(qq)) < qq).astype(int))
                      for _ in range(reps)])
        out["null"] = {"reps": reps, "mean_e": float(E.mean()),
                       "se": float(E.std(ddof=1) / np.sqrt(reps)),
                       "share_ge_threshold": float((E >= 1 / ALPHA).mean()), "max": float(E.max())}
        print(json.dumps(out["null"], indent=1))
    elif "--real" in sys.argv:
        p, qq, da, db, y = (q[c].to_numpy() for c in ("p", "q", "oa", "ob", "y"))
        out["primary"] = {"e_real": e_real(p, qq, da, db, y), "e_fair": e_fair(p, qq, y)}
        out["primary"]["reject"] = out["primary"]["e_real"] >= 1 / ALPHA
        out["quiet_filter_sensitivity"] = {}
        for days in (0, 30, 90):
            g = d[d["gap"] >= days]
            out["quiet_filter_sensitivity"][str(days)] = {"n": int(len(g)), "e_real": e_real(
                *(g[c].to_numpy() for c in ("p", "q", "oa", "ob", "y")))}
        out["by_year"] = {}
        for yr, g in q.groupby(q["dt"].dt.year):
            out["by_year"][str(yr)] = {"n": int(len(g)), "e_real": e_real(
                *(g[c].to_numpy() for c in ("p", "q", "oa", "ob", "y")))}
        out["flat_by_lambda"] = {}
        for lam in LAMBDAS:
            pb = sig(lam * logit(p) + (1 - lam) * logit(qq))
            ba, bb = pb * da > 1, (1 - pb) * db > 1
            k = ba | bb
            r = np.where(np.where(ba, y == 1, y == 0), np.where(ba, da, db) - 1, -1.0)[k]
            bs = [np.random.default_rng(i).choice(r, len(r)).mean() for i in range(2000)] if k.sum() >= 30 else None
            out["flat_by_lambda"][f"{lam:.2f}"] = {
                "bets": int(k.sum()), "roi": float(r.mean()) if k.any() else None,
                "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None}
        ll = lambda x: float(np.mean(-(y * np.log(np.clip(x, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - x, 1e-9, 1)))))  # noqa: E731
        out["log_loss"] = {"model": ll(p), "open": ll(qq),
                           "open_plus_model_0.25": ll(sig(0.25 * logit(p) + 0.75 * logit(qq)))}
        pr = out["primary"]
        print(f"PRIMARY  n {len(q):,}  real e {pr['e_real']:.4g}  (fair {pr['e_fair']:.4g})  → "
              f"{'REJECT' if pr['reject'] else 'no'} at e ≥ {1 / ALPHA:.0f}")
        print(json.dumps({k: out[k] for k in ("quiet_filter_sensitivity", "by_year", "log_loss")}, indent=1))
        for lam, v in out["flat_by_lambda"].items():
            roi = "—" if v["roi"] is None else f"{v['roi']:+.1%}"
            print(f"  λ {lam}: {v['bets']:5d} bets  ROI {roi}  {v['ci']}")
    else:
        print("sizes only; --null N to validate, --real to score")
    if "--json" in sys.argv:
        dst = Path(arg("--json", ""))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
