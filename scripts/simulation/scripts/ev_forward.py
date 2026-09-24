"""Pre-registration 4: the live test. One rule, frozen before the bouts it is
scored on have happened, scored at any checkpoint because the test is an
e-value.

  python3 scripts/ev_forward.py --label live-model --from 2026-09-25 --to <date> --null 1000
  python3 scripts/ev_forward.py --label live-model --from 2026-09-25 --to <date> --real \\
      --json results/forward_<date>.json

THE RULE (chosen on 2026's holdout after it was scored, which is why it has to
be tested on bouts that had not happened yet):
- p is the frozen model's probability for corner A;
- q is Bet365's opening price for corner A, the margin removed by the power
  method;
- p' = σ(0.10·logit p + 0.90·logit q);
- a side is bet where p' times its posted opening decimal is above 1.

THE BOUTS. Every bout after --from that Bet365 priced at the open, except a bout
where either fighter boxed between Bet365's opening timestamp and the bell: the
model would know that result, and the price would not.

THE TEST is about the rule's own bets. Each selected bet has a flat return r:
d − 1 if it wins, −1 if it loses. H0 says every selected bet has expected return
at most zero at the decimal Bet365 posted. Staking a fraction c of wealth on
each bet in turn multiplies wealth by 1 + c·r, and under H0 the expectation of
that is at most 1 for any c in [0, 1]. So the product is an e-value, mixed
uniformly over c ∈ {0.05, 0.10, …, 0.50}, and it may be read at any checkpoint
(Ville). It is rejected at e ≥ 20.

An earlier draft tested the same rule with Kelly stakes at λ = 0.10. At that
weight the blend sits so close to the price that Kelly bets almost nothing: on
2026's 39 bets it reached 1.7 while the bets themselves returned +37%. The test
above asks the question the rule asks. It is reported beside the Kelly version,
and the λ-mixture of the earlier protocols, neither of which decides anything.

`--real` is required to read outcomes.
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
CACHE = ROOT / "imports" / "staging"
import ev_window as W  # noqa: E402

ME = W.ME
LAMBDA = 0.10
ALPHA = 0.05
FEED = CACHE / "odds_external" / "betsapi.parquet"
BOARD = CACHE / "betsapi_board.parquet"


def load(label: str, t0: str, t1: str, with_y: bool) -> pd.DataFrame:
    ME.ODDS = FEED
    npz = np.load(CACHE / "preds" / f"{label}.npz", allow_pickle=True)
    pc = pd.Series(npz["p_corp"], index=npz["key_corp"])
    df, _ = ME.build(ME.arg("--tag", "l6"))
    j = ME.join_odds(df, verbose=False)
    same = ~j["swap"].to_numpy(bool)
    d = pd.DataFrame({"key": j["index"].to_numpy(), "dt": j["dt"].to_numpy(),
                      "oa": np.where(same, j["open_a"], j["open_b"]).astype(float),
                      "ob": np.where(same, j["open_b"], j["open_a"]).astype(float)})
    d = d[(d["dt"] > pd.Timestamp(t0)) & (d["dt"] <= pd.Timestamp(t1)) & (d["oa"] > 1)
          & (d["ob"] > 1) & d["key"].isin(pc.index)].copy()
    d["p"] = pc.loc[d["key"]].to_numpy()
    d["q"] = ME.devig(1 / d["oa"].to_numpy(), 1 / d["ob"].to_numpy(), "power")

    # Bet365's opening timestamp, re-attached by names and date
    feed, board = pd.read_parquet(FEED), pd.read_parquet(BOARD)
    ot = board[board["book"] == "Bet365"].groupby("event_id")["open_t"].min()
    feed["open_dt"] = pd.to_datetime(feed["event_id"].map(ot), unit="s")
    feed["dt"] = pd.to_datetime(feed["date"])
    key = lambda a, b: "|".join(sorted([ME.norm(a) or "", ME.norm(b) or ""]))  # noqa: E731
    feed["pair"] = [key(a, b) for a, b in zip(feed["a"], feed["b"])]
    names = df[["a_name", "b_name"]]
    d["pair"] = [key(a, b) for a, b in zip(names.loc[d["key"], "a_name"], names.loc[d["key"], "b_name"])]
    m = d[["pair", "dt"]].reset_index().merge(feed[["pair", "dt", "open_dt"]], on="pair", suffixes=("", "_f"))
    m = m[(m["dt"] - m["dt_f"]).abs().dt.days <= 1].drop_duplicates("index").set_index("index")
    d["open_dt"] = m["open_dt"].reindex(d.index)

    # a bout between the open and the bell, for either man
    sym = df[["dt", "a", "b"]]
    hist = pd.concat([sym[["dt", "a"]].rename(columns={"a": "f"}),
                      sym[["dt", "b"]].rename(columns={"b": "f"})])
    prev = {f: np.sort(g.values) for f, g in hist.groupby("f")["dt"]}

    def between(f, t0_, t1_):
        v = prev.get(f)
        return v is not None and not pd.isna(t0_) and bool(((v > t0_) & (v < t1_)).any())
    k = d["key"].to_numpy()
    d["intervening"] = [between(a, o, t) or between(b, o, t) for a, b, o, t in
                        zip(sym.loc[k, "a"].to_numpy(), sym.loc[k, "b"].to_numpy(),
                            d["open_dt"].to_numpy(), d["dt"].to_numpy())]
    d = d[~d["intervening"] & d["open_dt"].notna()]
    if with_y:
        yc = pd.Series(npz["y_corp"], index=npz["key_corp"])
        d["y"] = yc.loc[d["key"]].to_numpy().astype(int)
    return d.sort_values(["dt", "key"]).reset_index(drop=True)


FRACTIONS = np.round(np.arange(0.05, 0.501, 0.05), 2)


def flat_e(r: np.ndarray) -> float:
    """The mixture over c of prod(1 + c·r): an e-value against 'every bet has
    expected return at most zero at its posted decimal'."""
    if len(r) == 0:
        return 1.0
    logs = np.array([np.log(np.maximum(1 + c * r, 1e-12)).sum() for c in FRACTIONS])
    return float(np.exp(logs.max() + np.log(np.mean(np.exp(logs - logs.max())))))


def main() -> None:
    label, t0, t1 = W.arg("--label", ""), W.arg("--from", "2026-09-25"), W.arg("--to", "")
    real = "--real" in sys.argv
    d = load(label, t0, t1, with_y=real)
    p, q, da, db = (d[c].to_numpy() for c in ("p", "q", "oa", "ob"))
    pb = W.sig(LAMBDA * W.logit(p) + (1 - LAMBDA) * W.logit(q))
    ba, bb = pb * da > 1, (1 - pb) * db > 1
    k = ba | bb
    out: dict = {"label": label, "from": t0, "to": t1, "lambda": LAMBDA,
                 "bouts": int(len(d)), "bets": int(k.sum())}
    print(f"[{label}] {t0} → {t1}: {len(d):,} bouts, {int(k.sum()):,} bets at λ {LAMBDA}")
    side_d = np.where(ba, da, db)[k]
    if "--null" in sys.argv:
        # at the edge of H0: each backed side wins with probability exactly
        # 1/d, so every bet's expected return is zero and E[e] is exactly 1
        rng = np.random.default_rng(20260925)
        reps = int(W.arg("--null", "1000"))
        E = np.array([flat_e(np.where(rng.random(len(side_d)) < 1 / side_d, side_d - 1, -1.0))
                      for _ in range(reps)])
        out["null"] = {"reps": reps, "mean_e": float(E.mean()), "se": float(E.std(ddof=1) / np.sqrt(reps)),
                       "share_ge_20": float((E >= 1 / ALPHA).mean())}
        print(json.dumps(out["null"], indent=1))
    elif real:
        y = d["y"].to_numpy()
        r = np.where(np.where(ba, y == 1, y == 0), np.where(ba, da, db) - 1, -1.0)[k]
        e = flat_e(r)
        out["e_flat"], out["reject"] = e, e >= 1 / ALPHA
        out["e_kelly_lambda_0.10"] = float(np.exp(W.kelly_log(pb, da, db, y).sum()))
        out["flat"] = {"roi": float(r.mean()) if k.any() else None, "profit_units": float(r.sum())}
        if k.sum() >= 30:
            bs = [np.random.default_rng(i).choice(r, len(r)).mean() for i in range(2000)]
            out["flat"]["ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out["e_mixture"] = W.e_real(p, q, da, db, y)
        print(f"e = {e:.4g} → {'REJECT' if out['reject'] else 'not yet'} at e ≥ {1 / ALPHA:.0f} · "
              f"flat ROI {out['flat']['roi']} on {int(k.sum())} bets")
    if "--json" in sys.argv:
        dst = Path(W.arg("--json", ""))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
