"""THE edge test on the BoxRec regional tail (sourced from Supabase).

Same leak-free machinery as clv_test.py (chronological replay → Elo + SoS +
record → 3-outcome model, mirror-augmented, natural-orientation predict →
match test bouts to ProBoxingOdds closing lines → CLV on the competitive
subset), but the corpus is the BoxRec crawl now in the DB — i.e. the regional
tail where the soft-market edge, if it exists, should show up.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env.local")
ODDS = ROOT / "imports" / "staging" / "proboxingodds.parquet"
ELO_K, ELO_INIT = 32.0, 1500.0
FEATS = ["diff_elo", "diff_winrate", "diff_ko_rate", "diff_koed_rate",
         "diff_bouts", "diff_sos", "diff_layoff"]
_PAREN = re.compile(r"\[.*?\]|\(.*?\)")


def norm(s):
    if not isinstance(s, str) or not s.strip():
        return None
    s = _PAREN.sub("", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()) or None


def load_bouts() -> pd.DataFrame:
    import psycopg
    sql = """
      select fa.name_en as a, fb.name_en as b, e.date::date as dt,
             b.winner_id = b.fighter_a_id as a_won, b.is_draw, b.method
      from bout b
      join fighter fa on fa.id = b.fighter_a_id
      join fighter fb on fb.id = b.fighter_b_id
      join event e on e.id = b.event_id
      where b.status = 'completed' and e.date is not null
    """
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=20) as c, c.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    df["result"] = np.where(df["is_draw"], "draw", np.where(df["a_won"], "win", "loss"))
    return df.rename(columns={"a": "subject_name", "b": "opponent_name"})


def build():
    df = load_bouts()
    df["a"] = df["subject_name"].map(norm)
    df["b"] = df["opponent_name"].map(norm)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df[df.a.notna() & df.b.notna() & df.dt.notna() & (df.a != df.b)]
    df["method"] = df["method"].fillna("")

    def canon(r):
        a, b, res = r["a"], r["b"], r["result"]
        if a <= b:
            return pd.Series([a, b, 0 if res == "win" else (2 if res == "loss" else 1)])
        return pd.Series([b, a, 2 if res == "win" else (0 if res == "loss" else 1)])

    df[["fa", "fb", "outcome"]] = df.apply(canon, axis=1)
    df["key"] = df.fa + "|" + df.fb + "|" + df.dt.dt.strftime("%Y-%m-%d")
    df = df.sort_values("dt").drop_duplicates("key").reset_index(drop=True)
    print(f"deduped bouts: {len(df):,}  ({df.dt.min().date()} → {df.dt.max().date()})")

    elo = defaultdict(lambda: ELO_INIT)
    wins = defaultdict(int); n_ = defaultdict(int); kof = defaultdict(int); koa = defaultdict(int)
    opp = defaultdict(float); last = {}
    rows = []
    for r in df.itertuples(index=False):
        a, b, t = r.fa, r.fb, int(r.outcome)
        ea, eb, na, nb = elo[a], elo[b], n_[a], n_[b]
        rt = lambda d, k: (d / k) if k else np.nan
        rows.append({
            "diff_elo": ea - eb, "diff_winrate": rt(wins[a], na) - rt(wins[b], nb),
            "diff_ko_rate": rt(kof[a], na) - rt(kof[b], nb), "diff_koed_rate": rt(koa[a], na) - rt(koa[b], nb),
            "diff_bouts": na - nb, "diff_sos": rt(opp[a], na) - rt(opp[b], nb),
            "diff_layoff": ((r.dt - last[a]).days if a in last else np.nan) - ((r.dt - last[b]).days if b in last else np.nan),
            "both": int(na > 0 and nb > 0), "dt": r.dt, "target": t, "fa": a, "fb": b,
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
    fl = tr.copy(); fl[FEATS] = -fl[FEATS].values; fl["target"] = fl["target"].map({0: 2, 2: 0, 1: 1})
    tra = pd.concat([tr, fl], ignore_index=True)
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, log_loss
    params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "learning_rate": .03,
              "num_leaves": 31, "min_data_in_leaf": 80, "feature_fraction": .9, "bagging_fraction": .9,
              "bagging_freq": 5, "lambda_l2": 1., "verbosity": -1, "seed": 42}
    m = lgb.train(params, lgb.Dataset(tra[FEATS], label=tra["target"]), num_boost_round=700)
    P = m.predict(te[FEATS]); te["p_a"], te["p_b"] = P[:, 0], P[:, 2]
    print(f"train {len(tr):,} · test {len(te):,} ({te.dt.min().date()} → {te.dt.max().date()})")

    if not ODDS.exists():
        print("no odds parquet"); return
    o = pd.read_parquet(ODDS)
    o["na"] = o["a"].map(norm); o["nb"] = o["b"].map(norm); o["dd"] = pd.to_datetime(o["date"], errors="coerce")
    o = o[o.close_a.notna() & o.close_b.notna() & o.dd.notna() & o.na.notna() & o.nb.notna()].copy()

    def orient(r):
        ca, cb = (r.close_a, r.close_b) if r.na <= r.nb else (r.close_b, r.close_a)
        ia, ib = 1 / ca, 1 / cb
        return pd.Series([ia / (ia + ib), ca, cb])

    o[["mkt_pa", "cca", "ccb"]] = o.apply(orient, axis=1)
    o["pair"] = o.apply(lambda r: "|".join(sorted([r.na, r.nb])), axis=1)
    omap = {(r.pair, r.dd.date()): (r.mkt_pa, r.cca, r.ccb) for r in o.itertuples(index=False)}

    recs = []
    for r in te.itertuples(index=False):
        pair = "|".join(sorted([r.fa, r.fb]))
        for dd in range(-3, 4):
            k = (pair, (r.dt + pd.Timedelta(days=dd)).date())
            if k in omap:
                mp, ca, cb = omap[k]; recs.append((r.target, r.p_a, r.p_b, mp, ca, cb)); break
    mt = pd.DataFrame(recs, columns=["target", "p_a", "p_b", "mkt_pa", "cca", "ccb"])
    print(f"\nmatched to closing odds: {len(mt):,} test bouts")
    if len(mt) < 20:
        print("too few matches — need more overlap (crawl still expanding / odds vs tail coverage)"); return

    dec = mt[mt.target != 1].copy()
    dec["model_pa"] = dec.p_a / (dec.p_a + dec.p_b); dec["y"] = (dec.target == 0).astype(int)

    def roi_of(s):
        r = []
        for pm, pk, ca, cb, y in zip(s.model_pa.values, s.mkt_pa.values, s.cca.values, s.ccb.values, s.y.values):
            if pm > pk + .03:
                r.append((ca - 1) if y == 1 else -1)
            elif (1 - pm) > (1 - pk) + .03:
                r.append((cb - 1) if y == 0 else -1)
        return np.array(r)

    for name, lo, hi in (("ALL decisive", 0, 1), ("COMPETITIVE 30-70%", .30, .70)):
        s = dec[(dec.mkt_pa >= lo) & (dec.mkt_pa <= hi)]
        if len(s) < 10:
            print(f"  {name}: too few (n={len(s)})"); continue
        y = s.y.values
        mll = log_loss(y, s.model_pa.clip(.02, .98), labels=[0, 1]); kll = log_loss(y, s.mkt_pa.clip(.02, .98), labels=[0, 1])
        macc = accuracy_score(y, (s.model_pa > .5).astype(int)); kacc = accuracy_score(y, (s.mkt_pa > .5).astype(int))
        r = roi_of(s)
        if len(r):
            rng = np.random.default_rng(42)
            boots = [rng.choice(r, len(r), replace=True).mean() for _ in range(2000)]
            lo_, hi_ = np.percentile(boots, [2.5, 97.5]); pp = np.mean(np.array(boots) > 0)
            roi = f"{r.mean():+.1%} on {len(r)} bets [95% CI {lo_:+.1%}..{hi_:+.1%}, P>0={pp:.0%}]"
        else:
            roi = "no value bets"
        print(f"  {name} (n={len(s):,}): model LL {mll:.4f} acc {macc:.3f} | market LL {kll:.4f} acc {kacc:.3f}\n      close-line value ROI {roi}")


if __name__ == "__main__":
    main()
