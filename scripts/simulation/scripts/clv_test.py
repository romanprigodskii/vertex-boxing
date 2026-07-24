"""THE kill-test: does the model beat the closing line?

Leak-free replay → 3-outcome model (trained with mirror augmentation for
symmetry, predicted in natural orientation) → match test bouts to ProBoxingOdds
closing moneylines by name-pair + date → compare model vs market on the
COMPETITIVE subset (market-implied 30-70%), and a value-betting ROI at the close.

Honest expectation: these are notable/title fighters (efficient market), so
parity or a small loss is the likely, honest result — the soft-regional edge
needs the long tail. This proves the machinery and sets the baseline.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BOUTS = ROOT / "imports" / "staging" / "wikipedia_bouts.parquet"
SPINE = ROOT / "imports" / "staging" / "wikidata_boxers.parquet"
ODDS = ROOT / "imports" / "staging" / "proboxingodds.parquet"
ELO_K, ELO_INIT = 32.0, 1500.0
_PAREN = re.compile(r"\[.*?\]|\(.*?\)")
FEATS = ["diff_elo", "diff_winrate", "diff_ko_rate", "diff_koed_rate",
         "diff_bouts", "diff_sos", "diff_layoff", "diff_age"]


def norm(s):
    if not isinstance(s, str) or not s.strip():
        return None
    s = _PAREN.sub("", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()) or None


def build():
    df = pd.read_parquet(BOUTS)
    sp = pd.read_parquet(SPINE)
    dob = {}
    for r in sp.itertuples(index=False):
        n = norm(r.name)
        if n and isinstance(r.dob, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", r.dob):
            dob[n] = pd.Timestamp(r.dob)
    df["a"] = df["subject_name"].map(norm)
    df["b"] = df["opponent_name"].map(norm)
    df["dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df.a.notna() & df.b.notna() & df.dt.notna() & (df.a != df.b)]
    df = df[(df.dt >= "1889-01-01") & (df.dt <= "2026-12-31")]
    df = df[df.result.isin(["win", "loss", "draw"])]

    def canon(r):
        a, b, res = r["a"], r["b"], r["result"]
        if a <= b:
            return pd.Series([a, b, 0 if res == "win" else (2 if res == "loss" else 1)])
        return pd.Series([b, a, 2 if res == "win" else (0 if res == "loss" else 1)])

    df[["fa", "fb", "outcome"]] = df.apply(canon, axis=1)
    df["key"] = df.fa + "|" + df.fb + "|" + df.dt.dt.strftime("%Y-%m-%d")
    df = df.sort_values("dt").drop_duplicates("key").reset_index(drop=True)
    df["method"] = df["method"].fillna("")

    elo = defaultdict(lambda: ELO_INIT)
    wins = defaultdict(int); n_ = defaultdict(int); kof = defaultdict(int); koa = defaultdict(int)
    opp = defaultdict(float); last = {}
    rows = []
    for r in df.itertuples(index=False):
        a, b, t = r.fa, r.fb, int(r.outcome)
        ea, eb, na, nb = elo[a], elo[b], n_[a], n_[b]
        rt = lambda d, k: (d / k) if k else np.nan
        ag = lambda w: ((r.dt - dob[w]).days / 365.25 if w in dob else np.nan)
        rows.append({
            "diff_elo": ea - eb, "diff_winrate": rt(wins[a], na) - rt(wins[b], nb),
            "diff_ko_rate": rt(kof[a], na) - rt(kof[b], nb),
            "diff_koed_rate": rt(koa[a], na) - rt(koa[b], nb),
            "diff_bouts": na - nb, "diff_sos": rt(opp[a], na) - rt(opp[b], nb),
            "diff_layoff": ((r.dt - last[a]).days if a in last else np.nan) - ((r.dt - last[b]).days if b in last else np.nan),
            "diff_age": ag(a) - ag(b), "both": int(na > 0 and nb > 0),
            "dt": r.dt, "target": t, "fa": a, "fb": b,
        })
        sa = 1.0 if t == 0 else (0.5 if t == 1 else 0.0)
        exp = 1 / (1 + 10 ** ((eb - ea) / 400)); elo[a] = ea + ELO_K * (sa - exp); elo[b] = eb + ELO_K * ((1 - sa) - (1 - exp))
        opp[a] += eb; opp[b] += ea; n_[a] += 1; n_[b] += 1
        ko = any(k in r.method for k in ("ko", "tko", "rtd"))
        if t == 0:
            wins[a] += 1; kof[a] += ko; koa[b] += ko
        elif t == 2:
            wins[b] += 1; kof[b] += ko; koa[a] += ko
        last[a] = last[b] = r.dt
    d = pd.DataFrame(rows)
    return d[d["both"] == 1].reset_index(drop=True)


def main():
    d = build().sort_values("dt").reset_index(drop=True)
    n = len(d); tr = d.iloc[: int(n * .80)].copy(); te = d.iloc[int(n * .80):].copy()
    # mirror-augment train for symmetry
    fl = tr.copy(); fl[FEATS] = -fl[FEATS].values
    fl["target"] = fl["target"].map({0: 2, 2: 0, 1: 1})
    tra = pd.concat([tr, fl], ignore_index=True)
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, log_loss
    params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
              "learning_rate": .03, "num_leaves": 31, "min_data_in_leaf": 60,
              "feature_fraction": .9, "bagging_fraction": .9, "bagging_freq": 5, "lambda_l2": 1., "verbosity": -1, "seed": 42}
    m = lgb.train(params, lgb.Dataset(tra[FEATS], label=tra["target"]), num_boost_round=600)
    P = m.predict(te[FEATS])
    te["p_a"], te["p_dr"], te["p_b"] = P[:, 0], P[:, 1], P[:, 2]
    print(f"test bouts: {len(te):,}  ({te.dt.min().date()} → {te.dt.max().date()})")

    if not ODDS.exists():
        print("no odds yet — run run_proboxingodds.py first"); return
    o = pd.read_parquet(ODDS)
    o["na"] = o["a"].map(norm); o["nb"] = o["b"].map(norm); o["d"] = pd.to_datetime(o["date"], errors="coerce")
    o = o[o.close_a.notna() & o.close_b.notna() & o.d.notna()]
    # market prob for the (alphabetically-canonical) fighter_a
    o = o[o.na.notna() & o.nb.notna()].copy()
    o["pair"] = o.apply(lambda r: "|".join(sorted([r.na, r.nb])), axis=1)

    def orient(r):
        # canonical fighter_a = alphabetically first; carry the REAL closing
        # decimal odds (WITH vig) for each canonical side, plus the de-vigged
        # market prob. Winners must settle at the real odds, not 1/mkt_pa.
        ca, cb = (r.close_a, r.close_b) if r.na <= r.nb else (r.close_b, r.close_a)
        ia, ib = 1 / ca, 1 / cb
        return pd.Series([ia / (ia + ib), ca, cb])

    o[["mkt_pa", "close_ca", "close_cb"]] = o.apply(orient, axis=1)
    omap = {(r.pair, r.d.date()): (r.mkt_pa, r.close_ca, r.close_cb) for r in o.itertuples(index=False)}

    # match test bouts (±2 day tolerance)
    recs = []
    for r in te.itertuples(index=False):
        pair = "|".join(sorted([r.fa, r.fb]))
        for dd in range(-2, 3):
            key = (pair, (r.dt + pd.Timedelta(days=dd)).date())
            if key in omap:
                mp, cca, ccb = omap[key]
                recs.append((r.fa, r.fb, r.dt, r.target, r.p_a, r.p_b, mp, cca, ccb)); break
    mt = pd.DataFrame(recs, columns=["fa", "fb", "dt", "target", "p_a", "p_b", "mkt_pa", "close_ca", "close_cb"])
    print(f"matched to closing odds: {len(mt):,} test bouts")
    if len(mt) < 20:
        print("too few matches yet (odds still fetching) — re-run when it finishes"); return

    dec = mt[mt.target != 1].copy()  # decisive (moneyline is a 2-way market)
    dec["model_pa"] = dec.p_a / (dec.p_a + dec.p_b)
    dec["y"] = (dec.target == 0).astype(int)
    comp = dec[(dec.mkt_pa >= .30) & (dec.mkt_pa <= .70)].copy()
    print(f"\ndecisive matched: {len(dec):,} | competitive (mkt 30-70%): {len(comp):,}")
    for name, s in (("ALL decisive", dec), ("COMPETITIVE", comp)):
        if len(s) < 10:
            continue
        yy = s.y.values
        mll = log_loss(yy, s.model_pa.clip(.02, .98), labels=[0, 1])
        kll = log_loss(yy, s.mkt_pa.clip(.02, .98), labels=[0, 1])
        macc = accuracy_score(yy, (s.model_pa > .5).astype(int))
        kacc = accuracy_score(yy, (s.mkt_pa > .5).astype(int))
        # value betting at the close: bet the side the model favors over the
        # market by >3pp, settle at the REAL closing decimal odds (with vig) —
        # mkt_pa is used ONLY for value selection, never as the payout.
        roi = []
        for pa_m, pa_k, ca, cb, y in zip(
            s.model_pa.values, s.mkt_pa.values, s.close_ca.values, s.close_cb.values, s.y.values
        ):
            if pa_m > pa_k + .03:      # model likes A more than the market
                roi.append((ca - 1) if y == 1 else -1)
            elif (1 - pa_m) > (1 - pa_k) + .03:  # model likes B more
                roi.append((cb - 1) if y == 0 else -1)
        if roi:
            roi = np.array(roi)
            rng = np.random.default_rng(42)
            boots = [rng.choice(roi, len(roi), replace=True).mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            frac_pos = np.mean(np.array(boots) > 0)
            roi_str = f"{roi.mean():+.1%} on {len(roi)} bets [95% CI {lo:+.1%}..{hi:+.1%}, P(>0)={frac_pos:.0%}]"
        else:
            roi_str = "no value bets"
        print(f"  {name} (n={len(s):,}): model LL {mll:.4f} acc {macc:.3f} | market LL {kll:.4f} acc {kacc:.3f}\n      close-line value ROI {roi_str}")

    # ---- robustness stress-test (is the competitive edge real or cherry-picked?) ----
    def roi_of(s):
        r = []
        for pa_m, pa_k, ca, cb, y in zip(s.model_pa.values, s.mkt_pa.values, s.close_ca.values, s.close_cb.values, s.y.values):
            if pa_m > pa_k + .03:
                r.append((ca - 1) if y == 1 else -1)
            elif (1 - pa_m) > (1 - pa_k) + .03:
                r.append((cb - 1) if y == 0 else -1)
        return np.array(r)

    print("\n  ROBUSTNESS — value-bet ROI across probability bands:")
    for lo, hi in [(.20, .80), (.25, .75), (.30, .70), (.35, .65), (.40, .60)]:
        r = roi_of(dec[(dec.mkt_pa >= lo) & (dec.mkt_pa <= hi)])
        if len(r):
            rng = np.random.default_rng(1)
            fp = np.mean([rng.choice(r, len(r), replace=True).mean() for _ in range(1000)] > np.float64(0))
            print(f"    band {lo:.0%}-{hi:.0%}: ROI {r.mean():+.1%} on {len(r)} bets (P>0={fp:.0%})")
    print("  ROBUSTNESS — competitive (30-70%) ROI by test year:")
    comp = comp.copy(); comp["yr"] = pd.to_datetime(comp.dt).dt.year
    for yr, g in comp.groupby("yr"):
        r = roi_of(g)
        if len(r) >= 8:
            print(f"    {yr}: ROI {r.mean():+.1%} on {len(r)} bets | model acc {(g.model_pa>.5).eq(g.y).mean():.3f} vs mkt {(g.mkt_pa>.5).eq(g.y).mean():.3f}")


if __name__ == "__main__":
    main()
