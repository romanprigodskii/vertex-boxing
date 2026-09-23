"""What the e-value audit's two money results turned out to be. Exploratory: none
of this was pre-registered, and every check here was chosen after the audit ran.

The audit (ev_audit.py, docs/evalue_protocol.md) rejected P3 (money at Bet365's
open) and P4 (money at its close) on the published model. Both looked too good,
so each got the checks that could kill it:

  P4, the close. Split by which snapshot the close came from. On the bouts whose
     close is a true `kickoff` price, the real e-value is ~1. The rejection lives
     on bouts where Bet365 ran no in-play market, whose "close" is the last price
     it posted and can be days old. At ProBoxingOdds' close it is below 80.
     Verdict: a stale-price artifact, not money at the close.

  P3, the open. The published model reads the weigh-in, the referee, the card's
     judges and the running order, none of which exists when the line opens. So:
       - retrain without them (`market_eval --drop weigh+ref+judc+cardpos`,
         label openinfo-close);
       - the real process under a simulated null;
       - junk opening books;
       - by year;
       - a fight in between the open and the bout;
       - price-only controls with no model in them;
       - the same bets at ProBoxingOdds' open;
       - flat stakes over the registry's whole λ grid.

  python3 scripts/ev_followup.py --json results/evalue_followup.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(ROOT / "scripts" / "simulation" / "scripts"))
import ev_audit as A  # noqa: E402
import ev_registry as R  # noqa: E402
import market_eval as ME  # noqa: E402

CACHE = ROOT / "imports" / "staging"


def boot(x, n=2000):
    b = [np.random.default_rng(i).choice(x, len(x)).mean() for i in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def mixture(p, q, lams=R.LAMBDAS):
    return [A.sig(lam * A.logit(p) + (1 - lam) * A.logit(q)) for lam in lams]


def real(alts, da, db, y):
    return float(A.mix([A.kelly_log(a, da, db, y) for a in alts])[-1])


def fair(alts, q, y):
    return float(A.mix([A.lr_log(a, q, y) for a in alts])[-1])


def flat(p, q, da, db, y, lam):
    """1-unit bets wherever the λ-blend prices a side above the posted decimal."""
    pb = A.sig(lam * A.logit(p) + (1 - lam) * A.logit(q))
    ba, bb = pb * da > 1, (1 - pb) * db > 1
    k = ba | bb
    r = np.where(np.where(ba, y == 1, y == 0), np.where(ba, da, db) - 1, -1.0)[k]
    return {"bets": int(k.sum()), "roi": float(r.mean()) if k.any() else None,
            "ci": boot(r) if k.sum() >= 30 else None}, ba, bb, k


def attach_feed(c: pd.DataFrame) -> pd.DataFrame:
    """Close snapshot and Bet365's opening timestamp, re-attached by names and date
    (join_odds carries prices only)."""
    feed = pd.read_parquet(A.FEED)
    board = pd.read_parquet(CACHE / "betsapi_board.parquet")
    ot = board[board["book"] == "Bet365"].groupby("event_id")["open_t"].min()
    feed["open_dt"] = pd.to_datetime(feed["event_id"].map(ot), unit="s")
    feed["dt"] = pd.to_datetime(feed["date"])
    key = lambda a, b: "|".join(sorted([ME.norm(a) or "", ME.norm(b) or ""]))  # noqa: E731
    feed["pair"] = [key(a, b) for a, b in zip(feed["a"], feed["b"])]
    sym = pd.read_parquet(CACHE / "sym_l6.parquet", columns=["a_name", "b_name"])
    c["pair"] = [key(a, b) for a, b in zip(sym.loc[c["key"], "a_name"], sym.loc[c["key"], "b_name"])]
    m = c[["pair", "dt"]].reset_index().merge(
        feed[["pair", "dt", "close_from", "open_dt"]], on="pair", suffixes=("", "_f"))
    m = m[(m["dt"] - m["dt_f"]).abs().dt.days <= 1].drop_duplicates("index").set_index("index")
    c["close_from"] = m["close_from"].reindex(c.index).fillna("alias-joined")
    c["open_dt"] = m["open_dt"].reindex(c.index)
    return c


def main() -> None:
    out: dict = {"note": "exploratory; chosen after the pre-registered audit ran"}
    d = A.load()
    c = attach_feed(d[d["dt"] <= R.CONFIRM[1]].reset_index(drop=True))
    y, q, ca, cb = c["y"].to_numpy(), c["q"].to_numpy(), c["ca"].to_numpy(), c["cb"].to_numpy()
    p_pub = c["p"].to_numpy()
    o = np.load(CACHE / "preds" / "openinfo-close.npz", allow_pickle=True)
    p_safe = pd.Series(o["p_corp"], index=o["key_corp"]).reindex(c["key"]).to_numpy()

    # --- P4: the close, by snapshot
    out["close_by_snapshot"] = {}
    for src in ["kickoff", "end", "start", "alias-joined"]:
        m = (c["close_from"] == src).to_numpy()
        if m.sum() >= 30:
            out["close_by_snapshot"][src] = {
                "n": int(m.sum()), "e_fair": fair(mixture(p_pub[m], q[m]), q[m], y[m]),
                "e_real": real(mixture(p_pub[m], q[m]), ca[m], cb[m], y[m])}

    # --- P3: the open
    ho = c["q_open"].notna().to_numpy()
    y3, qo, oa, ob = y[ho], c["q_open"].to_numpy()[ho], c["oa"].to_numpy()[ho], c["ob"].to_numpy()[ho]
    ps, pp = p_safe[ho], p_pub[ho]
    ov = 1 / oa + 1 / ob
    out["open"] = {
        "n": int(ho.sum()), "overround_median": float(np.median(ov) - 1),
        "books_under_100pct": int((ov < 1).sum()),
        "published_model": {"e_fair": fair(mixture(pp, qo), qo, y3), "e_real": real(mixture(pp, qo), oa, ob, y3)},
        "open_safe_model": {"e_fair": fair(mixture(ps, qo), qo, y3), "e_real": real(mixture(ps, qo), oa, ob, y3)},
        "controls_no_model": {
            "price_recalibrated": real([A.sig(b * A.logit(qo)) for b in R.BETAS], oa, ob, y3),
            "favourite_underpriced": real([A.sig(A.logit(qo) + np.where(qo >= .5, 1, -1) * dl)
                                           for dl in R.DELTAS], oa, ob, y3)},
    }
    fav = qo >= .5
    r_fav = np.where(fav, np.where(y3 == 1, oa - 1, -1), np.where(y3 == 0, ob - 1, -1))
    out["open"]["every_favourite_flat"] = {"roi": float(r_fav.mean()), "ci": boot(r_fav)}

    rng = np.random.default_rng(7)
    E = np.array([real(mixture(ps, qo), oa, ob, (rng.random(len(qo)) < qo).astype(int))
                  for _ in range(400)])
    out["open"]["null_real"] = {"reps": 400, "mean_e": float(E.mean()),
                                "share_ge_80": float((E >= 80).mean()), "max": float(E.max())}
    dts = c["dt"].to_numpy()[ho]
    out["open"]["by_year"] = {str(yr): real(mixture(ps[m], qo[m]), oa[m], ob[m], y3[m])
                              for yr in (2023, 2024, 2025)
                              for m in [pd.DatetimeIndex(dts).year == yr]}
    lead = (c["dt"] - c["open_dt"]).dt.days.to_numpy()[ho]
    out["open"]["days_open_to_bout"] = {"median": float(np.nanmedian(lead)),
                                        "p90": float(np.nanpercentile(lead, 90))}

    sym = pd.read_parquet(CACHE / "sym_l6.parquet", columns=["dt", "a", "b"])
    hist = pd.concat([sym[["dt", "a"]].rename(columns={"a": "f"}),
                      sym[["dt", "b"]].rename(columns={"b": "f"})])
    byf = hist.groupby("f")["dt"].apply(lambda s: np.sort(s.values))
    keys, odt = c["key"].to_numpy()[ho], c["open_dt"].to_numpy()[ho]

    def fought_between(f, t0, t1):
        if pd.isna(t0) or f not in byf.index:
            return False
        v = byf[f]
        return bool(((v > t0) & (v < t1)).any())
    inb = np.array([fought_between(sym.at[k, "a"], odt[i], dts[i]) or
                    fought_between(sym.at[k, "b"], odt[i], dts[i]) for i, k in enumerate(keys)])
    clean = ~pd.isna(odt) & ~inb
    out["open"]["fight_in_between"] = {
        "n": int(inb.sum()),
        "e_real_without": real(mixture(ps[clean], qo[clean]), oa[clean], ob[clean], y3[clean])}

    out["open"]["flat_by_lambda"] = {}
    for lam in R.LAMBDAS:
        f, ba, bb, k = flat(ps, qo, oa, ob, y3, lam)
        out["open"]["flat_by_lambda"][f"{lam:.2f}"] = f
    _, ba, bb, k = flat(ps, qo, oa, ob, y3, 0.25)
    sq_o, sq_c = np.where(ba, qo, 1 - qo)[k], np.where(ba, q[ho], 1 - q[ho])[k]
    out["open"]["clv_of_lambda_0.25_bets"] = {"moved_towards": float(np.mean(sq_c > sq_o)),
                                              "mean": float(np.mean(sq_c - sq_o))}

    # --- the forecasts side by side, log-loss on the same bouts: how far the
    # model moves the opening price towards the closing one
    def ll(x):
        x = np.clip(x, 1e-9, 1 - 1e-9)
        return float(np.mean(-(y3 * np.log(x) + (1 - y3) * np.log(1 - x))))
    qc = q[ho]
    out["log_loss"] = {"model": ll(ps), "open": ll(qo),
                       **{f"open_plus_model_{lam:.2f}": ll(A.sig(lam * A.logit(ps) + (1 - lam) * A.logit(qo)))
                          for lam in (0.15, 0.25, 0.35)},
                       "close": ll(qc),
                       "close_plus_model_0.15": ll(A.sig(0.15 * A.logit(ps) + 0.85 * A.logit(qc)))}

    # --- the same at ProBoxingOdds' open: another source, the same window
    pdts = pd.read_parquet(CACHE / "sym_l6.parquet", columns=["dt"]).iloc[o["key"]]["dt"].to_numpy()
    ok = (np.isfinite(o["oa"]) & np.isfinite(o["ob"]) & (o["oa"] > 1) & (o["ob"] > 1)
          & (pdts <= np.datetime64(R.CONFIRM[1])))
    py, poa, pob, pp_ = o["y"][ok], o["oa"][ok], o["ob"][ok], o["p"][ok]
    pqo = ME.devig(1 / poa, 1 / pob, "power")
    out["proboxingodds_open"] = {"n": int(ok.sum()),
                                 "e_real": real(mixture(pp_, pqo), poa, pob, py),
                                 "flat_lambda_0.25": flat(pp_, pqo, poa, pob, py, 0.25)[0]}

    print(json.dumps(out, indent=1))
    if "--json" in sys.argv:
        Path(ME.arg("--json", "")).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
