"""The e-value audit of docs/evalue_protocol.md: the published model, and a list of
price biases, bet against Bet365's closing line under a registry fixed first.

  python3 scripts/ev_audit.py --null 2000          # validation: outcomes drawn from the price
  python3 scripts/ev_audit.py --json results/evalue_audit.json    # the audit

THE NULL. For each bout, q is Bet365's closing price for corner A with the margin
taken out by the power method, which the protocol fixes in advance (it is the
method that calibrates the published run's training slice, slope 1.02). H0: q is
the probability that A wins, given everything known before the bell. Draws are
void, so the null is about the two-way outcome.

THE BET. Against an alternative probability p', full Kelly at fair odds buys the
increment p'(y)/q(y): E_H0 = 1, so the product over a slice is a test martingale,
its value at any stopping time an e-value (Ville). Its log is the slice's
log-loss improvement of p' over q, times n — the scoreboard's own quantity in the
currency of a bet. The alternative is a MIXTURE over a grid fixed in the registry
(blend weights for M, logit shifts for B), so no weight is chosen after the fact:
the mixture of martingales is a martingale.

THE REAL COLUMN. The same alternatives, staked by Kelly at the decimals Bet365
actually posted, which bets only where p' times the price exceeds 1. Every bet at
posted prices has E ≤ 1 for any true probability inside the band the two prices
leave (1 - 1/d_b, 1/d_a), so this is an e-value against that composite null with
nothing de-vigged in it — the null a bettor faces.

Stakes do not depend on earlier outcomes (p and q are fixed before the bell and
the grids are fixed), so the terminal e-value does not depend on the order the
bouts are multiplied in; running paths are drawn in date order.
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
FEED = CACHE / "odds_external" / "betsapi.parquet"
BASE = CACHE / "proboxingodds_v2.parquet"
os.environ["VERTEX_ODDS"] = str(FEED)
import market_eval as ME  # noqa: E402
import ev_registry as R  # noqa: E402

EPS = 1e-12


def sig(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ instruments
def lr_log(pa, q, y):
    """log of p'(y)/q(y) per bout: full Kelly at fair odds."""
    pa, q = np.clip(pa, 1e-9, 1 - 1e-9), np.clip(q, 1e-9, 1 - 1e-9)
    return np.where(y == 1, np.log(pa) - np.log(q), np.log(1 - pa) - np.log(1 - q))


def kelly_log(pa, da, db, y):
    """log wealth increment of Kelly at the posted decimals: back A if p'·d_a > 1,
    back B if (1-p')·d_b > 1, neither otherwise (both cannot hold on a book whose
    prices imply 100% or more)."""
    f = np.clip((pa * da - 1) / np.maximum(da - 1, 1e-9), 0, 1)
    g = np.clip(((1 - pa) * db - 1) / np.maximum(db - 1, 1e-9), 0, 1)
    ok = np.isfinite(da) & np.isfinite(db)
    f, g = np.where(ok, f, 0), np.where(ok, g, 0)
    m = np.where(y == 1, 1 + f * (da - 1) - g, 1 - f + g * (db - 1))
    return np.log(np.maximum(m, EPS))


def mix(logs: list[np.ndarray]) -> np.ndarray:
    """Running e-value of the uniform mixture of several processes, in date order."""
    cum = np.stack([np.cumsum(x) for x in logs])            # (components, n)
    top = cum.max(axis=0)
    return np.exp(top + np.log(np.mean(np.exp(cum - top), axis=0)))


def alternatives(h, d):
    """The alternative probabilities for corner A, one per mixture component."""
    if h.kind == "M":
        return [sig(lam * logit(d["p"]) + (1 - lam) * logit(d["q"])) for lam in R.LAMBDAS]
    side = h.side(d).to_numpy(float)
    qs = np.where(side == 1, d["q"], 1 - d["q"])
    signs = [h.sign] if h.sign else [+1, -1]
    out = []
    for s in signs:
        for dl in R.DELTAS:
            ps = sig(logit(qs) + s * dl)
            out.append(np.where(side == 1, ps, 1 - ps))
    return out


def score(alts, q, y, da, db) -> dict:
    n = len(y)
    if n == 0:
        return {"n": 0, "e_fair": 1.0, "e_real": 1.0}
    fair = mix([lr_log(a, q, y) for a in alts])
    real = mix([kelly_log(a, da, db, y) for a in alts])
    return {"n": int(n), "e_fair": float(fair[-1]), "e_real": float(real[-1]),
            "max_fair": float(fair.max()), "max_real": float(real.max()),
            "growth_fair": float(np.log(max(fair[-1], EPS)) / n),
            "growth_real": float(np.log(max(real[-1], EPS)) / n),
            "path_fair": fair, "path_real": real}


def e_bh(e: np.ndarray, alpha: float) -> np.ndarray:
    """Wang & Ramdas (2022): reject the k largest where e_(k) >= K/(αk).
    FDR <= α under arbitrary dependence, which is the only kind these have."""
    K = len(e)
    order = np.argsort(-e)
    k_star = 0
    for k in range(K, 0, -1):
        if e[order[k - 1]] >= K / (alpha * k):
            k_star = k
            break
    rej = np.zeros(K, bool)
    rej[order[:k_star]] = True
    return rej


# ------------------------------------------------------------------ the data
def load(tag: str = "l6") -> pd.DataFrame:
    """Every Bet365-priced bout in the published model's holdout, one row each,
    with the model's p, both prices and the registry's covariates."""
    df, _ = ME.build(tag)
    j = ME.join_odds(df, verbose=False)
    same = ~j["swap"].to_numpy(bool)
    pick = lambda x, z: np.where(same, j[x], j[z]).astype(float)  # noqa: E731
    d = pd.DataFrame({"key": j["index"].to_numpy(), "dt": j["dt"].to_numpy(),
                      "ca": pick("close_a", "close_b"), "cb": pick("close_b", "close_a"),
                      "oa": pick("open_a", "open_b"), "ob": pick("open_b", "open_a")})
    d = d[(d["ca"] > 1) & (d["cb"] > 1)]
    npz = np.load(CACHE / "preds" / "final-close.npz", allow_pickle=True)
    pc = pd.Series(npz["p_corp"], index=npz["key_corp"])
    yc = pd.Series(npz["y_corp"], index=npz["key_corp"])
    d = d[d["key"].isin(pc.index) & (d["dt"] > R.CONFIRM[0])].copy()
    d["p"] = pc.loc[d["key"]].to_numpy()
    d["y"] = yc.loc[d["key"]].to_numpy().astype(int)
    d["q"] = ME.devig(1 / d["ca"].to_numpy(), 1 / d["cb"].to_numpy(), "power")
    has_o = (d["oa"] > 1) & (d["ob"] > 1)
    d["q_open"] = np.nan
    d.loc[has_o, "q_open"] = ME.devig(1 / d.loc[has_o, "oa"].to_numpy(),
                                      1 / d.loc[has_o, "ob"].to_numpy(), "power")
    # P2's population: bouts the published feed does not price on any date
    ME.ODDS = BASE
    base = set(ME.join_odds(df, verbose=False)["index"])
    ME.ODDS = FEED
    d["fresh"] = ~d["key"].isin(base)
    cov = R.covariates(tag)
    d = d.join(cov.drop(columns=["dt"]), on="key")
    return d.sort_values(["dt", "key"]).reset_index(drop=True)


def run(d: pd.DataFrame, y: np.ndarray, keep_paths: bool = False) -> dict:
    q, ca, cb = d["q"].to_numpy(), d["ca"].to_numpy(), d["cb"].to_numpy()
    out: dict = {"primary": {}, "family": {}}

    def m_alts(pp, qq):
        return [sig(lam * logit(pp) + (1 - lam) * logit(qq)) for lam in R.LAMBDAS]

    allm = np.ones(len(d), bool)
    out["primary"]["P1"] = score(m_alts(d["p"].to_numpy(), q), q, y, ca, cb)
    fr = d["fresh"].to_numpy()
    out["primary"]["P2"] = score(m_alts(d["p"].to_numpy()[fr], q[fr]), q[fr], y[fr], ca[fr], cb[fr])
    ho = d["q_open"].notna().to_numpy()
    qo = d["q_open"].to_numpy()[ho]
    s3 = score(m_alts(d["p"].to_numpy()[ho], qo), qo, y[ho],
               d["oa"].to_numpy()[ho], d["ob"].to_numpy()[ho])
    out["primary"]["P3"] = s3                     # decided on e_real, at the open
    out["primary"]["P4"] = out["primary"]["P1"]   # decided on e_real, at the close
    betas = [sig(b * logit(q)) for b in R.BETAS]
    out["control"] = {"C1": score(betas, q, y, ca, cb)}
    del allm

    for h in R.FAMILY:
        m = h.mask(d).fillna(False).to_numpy(bool)
        if h.kind == "B":
            m &= h.side(d).notna().to_numpy()
        sub = d[m]
        out["family"][h.hid] = score(alternatives(h, sub), q[m], y[m], ca[m], cb[m])
        out["family"][h.hid]["name"] = h.name

    # --- the ladder and the sensitivities of protocol section 7: reported,
    # never decided on
    p_ = d["p"].to_numpy()
    fixed = [sig(0.17 * logit(p_) + 0.83 * logit(q))]
    qp = ME.devig(1 / ca, 1 / cb, "proportional")
    sens = {"P1_fixed_lambda_0.17": score(fixed, q, y, ca, cb),
            "P1_proportional_devig": score(m_alts(p_, qp), qp, y, ca, cb)}
    npz = np.load(CACHE / "preds" / "final-close.npz", allow_pickle=True)
    pbo_q = pd.Series(npz["p_mkt"], index=npz["key"])
    sh = d["key"].isin(pbo_q.index).to_numpy()
    if sh.any():
        qb = pbo_q.loc[d["key"][sh]].to_numpy()
        sens["shared_betsapi_close"] = score(m_alts(p_[sh], q[sh]), q[sh], y[sh], ca[sh], cb[sh])
        sens["shared_proboxingodds_close"] = score(m_alts(p_[sh], qb), qb, y[sh],
                                                   np.full(sh.sum(), np.nan), np.full(sh.sum(), np.nan))
    out["sensitivity"] = sens
    for blk in (out["primary"], out["control"], out["family"], sens):
        for v in blk.values():
            g = v.get("growth_fair", 0)
            v["bouts_to_1_over_alpha_fair"] = float(np.log(1 / R.ALPHA) / g) if g > 0 else None

    thr = 4 / R.ALPHA
    out["primary_decision"] = {
        "threshold": thr,
        "P1": out["primary"]["P1"]["e_fair"] >= thr,
        "P2": out["primary"]["P2"]["e_fair"] >= thr,
        "P3": out["primary"]["P3"]["e_real"] >= thr,
        "P4": out["primary"]["P4"]["e_real"] >= thr,
    }
    ids = [h.hid for h in R.FAMILY]
    ef = np.array([out["family"][i]["e_fair"] for i in ids])
    er = np.array([out["family"][i]["e_real"] for i in ids])
    out["ebh_fair"] = [i for i, r in zip(ids, e_bh(ef, R.ALPHA)) if r]
    out["ebh_real"] = [i for i, r in zip(ids, e_bh(er, R.ALPHA)) if r]
    if not keep_paths:
        for blk in (out["primary"], out["control"], out["family"], out["sensitivity"]):
            for v in blk.values():
                v.pop("path_fair", None)
                v.pop("path_real", None)
    return out


def null_check(d: pd.DataFrame, reps: int) -> dict:
    """Outcomes drawn from q itself, prices and stakes held: every e-value should
    average ≤ 1, cross 1/α at most α of the time, and e-BH should reject
    anything at most α of the time. Reads no outcome."""
    rng = np.random.default_rng(20260923)
    q = d["q"].to_numpy()
    ids = [h.hid for h in R.FAMILY]
    E = np.zeros((reps, len(ids)))
    P1 = np.zeros(reps)
    anyrej = 0
    for r in range(reps):
        y = (rng.random(len(q)) < q).astype(int)
        o = run(d, y)
        E[r] = [o["family"][i]["e_fair"] for i in ids]
        P1[r] = o["primary"]["P1"]["e_fair"]
        anyrej += bool(o["ebh_fair"])
    # How many of the K cross 1/α when nothing is there. The naive yardstick is
    # 5% of K, but the slices overlap and move together, so the count's spread
    # under the null is wider than a binomial's — and only a simulation of this
    # pool says how wide. The real count is read against THIS distribution.
    cnt = (E >= 1 / R.ALPHA).sum(axis=1)
    # per hypothesis: the mean, its Monte Carlo standard error, and the mean with
    # the single largest replicate left out — an e-value's mean over simulations
    # is carried by its rare large draws, so "within Monte Carlo error of 1" has
    # to be read against the error, not against 1
    se = E.std(axis=0, ddof=1) / np.sqrt(reps)
    loo = (E.sum(axis=0) - E.max(axis=0)) / (reps - 1)
    per = {i: {"mean": float(E[:, k].mean()), "se": float(se[k]),
               "z": float((E[:, k].mean() - 1) / se[k]) if se[k] > 0 else 0.0,
               "mean_without_largest": float(loo[k]), "largest": float(E[:, k].max())}
           for k, i in enumerate(ids)}
    return {"reps": reps, "mean_e_family": float(E.mean()), "per_hypothesis": per,
            "count_e_ge_20": {"mean": float(cnt.mean()), "p95": float(np.percentile(cnt, 95)),
                              "p99": float(np.percentile(cnt, 99)),
                              "hist": np.bincount(cnt).tolist()},
            "max_mean_e_single": float(E.mean(axis=0).max()),
            "share_e_ge_20": float((E >= 1 / R.ALPHA).mean()),
            "max_share_e_ge_20_single": float((E >= 1 / R.ALPHA).mean(axis=0).max()),
            "ebh_any_rejection": anyrej / reps, "mean_e_P1": float(P1.mean())}


def main() -> None:
    d = load(ME.arg("--tag", "l6"))
    conf = d[d["dt"] <= R.CONFIRM[1]].reset_index(drop=True)
    print(f"{len(d):,} Bet365-priced holdout bouts · confirmatory {len(conf):,} "
          f"({conf['dt'].min().date()} → {conf['dt'].max().date()}) · "
          f"fresh {int(conf['fresh'].sum()):,} · with an open {int(conf['q_open'].notna().sum()):,}")
    sizes = {h.hid: int((h.mask(conf).fillna(False)
                         & (h.side(conf).notna() if h.kind == "B" else True)).sum())
             for h in R.FAMILY}
    print("slice sizes:", sizes)

    if "--null" in sys.argv:
        res = null_check(conf, int(ME.arg("--null", "1000")))
        print(json.dumps(res, indent=1))
        if "--json" in sys.argv:
            Path(ME.arg("--json", "")).write_text(json.dumps(res, indent=1) + "\n")
        return

    null_file = ROOT / "scripts" / "simulation" / "results" / "evalue_null.json"
    out = {"confirmatory": run(conf, conf["y"].to_numpy()),
           "with_seen_2026": run(d, d["y"].to_numpy()),
           "n_confirmatory": int(len(conf)), "n_all": int(len(d)),
           "slice_sizes": sizes, "K": R.K, "alpha": R.ALPHA}
    c = out["confirmatory"]
    print("\nPRIMARY (e ≥ 80 rejects)")
    for k in R.PRIMARY:
        v = c["primary"][k]
        print(f"  {k}  n {v['n']:5,d}  fair {v['e_fair']:.3g}  real {v['e_real']:.3g}  "
              f"→ {'REJECT' if c['primary_decision'][k] else 'no'}")
    v = c["control"]["C1"]
    print(f"  C1  n {v['n']:5,d}  fair {v['e_fair']:.3g}  (control, no decision)")
    print(f"\nFAMILY K={R.K}, e-BH α={R.ALPHA}: fair rejects {c['ebh_fair'] or 'none'} · "
          f"real rejects {c['ebh_real'] or 'none'}")
    n20 = sum(v["e_fair"] >= 1 / R.ALPHA for v in c["family"].values())
    c["count_e_ge_20"] = int(n20)
    if null_file.exists():
        nl = json.loads(null_file.read_text())["count_e_ge_20"]
        hist = np.array(nl["hist"], float)
        p_count = float(hist[n20:].sum() / hist.sum()) if n20 < len(hist) else 0.0
        c["count_vs_null"] = {"null_mean": nl["mean"], "null_p95": nl["p95"], "p": p_count}
        print(f"  slices at e ≥ 20: {n20} of {R.K}; under the simulated null the count "
              f"averages {nl['mean']:.2f}, 95th pct {nl['p95']:.0f} · P(≥ {n20} | null) = {p_count:.3f}")
    for h in R.FAMILY:
        v = c["family"][h.hid]
        print(f"  {h.hid} {v['n']:5,d}  fair {v['e_fair']:9.3g}  real {v['e_real']:9.3g}  {h.name}")
    if "--json" in sys.argv:
        dst = Path(ME.arg("--json", ""))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
