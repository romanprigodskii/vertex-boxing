"""Proper leak-free 3-outcome model on the Wikipedia record corpus.

Improves on prototype_openboxing.py: (1) real corpus (full careers of ~10.7k
notable boxers, not just title bouts), (2) A/B SYMMETRIZATION so the model
can't cheat off slot order, (3) name-based identity with mirror-row dedup so a
bout isn't replayed twice, (4) age from the Wikidata DOB spine.

Reads the staged parquet as-it-grows, so it runs on the partial pull too.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BOUTS = ROOT / "imports" / "staging" / "wikipedia_bouts.parquet"
SPINE = ROOT / "imports" / "staging" / "wikidata_boxers.parquet"

ELO_K, ELO_INIT = 32.0, 1500.0
_PAREN = re.compile(r"\[.*?\]|\(.*?\)")


def norm(s) -> str | None:
    if not isinstance(s, str) or not s.strip():
        return None
    s = _PAREN.sub("", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return " ".join(s.split()) or None


def flip(key: str) -> bool:
    return hashlib.blake2b(key.encode(), digest_size=4).digest()[-1] % 2 == 1


def main() -> None:
    df = pd.read_parquet(BOUTS)
    spine = pd.read_parquet(SPINE)
    dob = {}
    for r in spine.itertuples(index=False):
        n = norm(r.name)
        if n and isinstance(r.dob, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", r.dob):
            dob[n] = pd.Timestamp(r.dob)

    df["a"] = df["subject_name"].map(norm)
    df["b"] = df["opponent_name"].map(norm)
    df["dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["a"].notna() & df["b"].notna() & df["dt"].notna() & (df["a"] != df["b"])]
    # sanity-filter parse noise (stray far-future/ancient dates from odd tables)
    df = df[(df["dt"] >= "1889-01-01") & (df["dt"] <= "2026-12-31")]
    df = df[df["result"].isin(["win", "loss", "draw"])]

    # canonical dedup: one row per {pair, date}, oriented a<b, outcome remapped
    def canon(row):
        a, b, res = row["a"], row["b"], row["result"]
        if a <= b:
            out = 0 if res == "win" else (2 if res == "loss" else 1)
            return pd.Series([a, b, out])
        else:
            out = 2 if res == "win" else (0 if res == "loss" else 1)
            return pd.Series([b, a, out])

    df[["fa", "fb", "outcome"]] = df.apply(canon, axis=1)
    df["key"] = df["fa"] + "|" + df["fb"] + "|" + df["dt"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("dt").drop_duplicates("key").reset_index(drop=True)
    df["method"] = df["method"].fillna("")
    print(f"deduped bouts: {len(df):,}  ({df['dt'].min().date()} → {df['dt'].max().date()})")

    elo = defaultdict(lambda: ELO_INIT)
    wins = defaultdict(int); losses = defaultdict(int); draws = defaultdict(int)
    kof = defaultdict(int); koa = defaultdict(int); n_ = defaultdict(int)
    opp_sum = defaultdict(float); last = {}

    rows = []
    for r in df.itertuples(index=False):
        a, b, t = r.fa, r.fb, int(r.outcome)
        ea, eb, na, nb = elo[a], elo[b], n_[a], n_[b]
        rt = lambda d, n: (d / n) if n else np.nan
        ag = lambda who: ((r.dt - dob[who]).days / 365.25 if who in dob else np.nan)
        rows.append({
            "diff_elo": ea - eb,
            "diff_winrate": rt(wins[a], na) - rt(wins[b], nb),
            "diff_ko_rate": rt(kof[a], na) - rt(kof[b], nb),
            "diff_koed_rate": rt(koa[a], na) - rt(koa[b], nb),
            "diff_bouts": na - nb,
            "diff_sos": rt(opp_sum[a], na) - rt(opp_sum[b], nb),
            "diff_layoff": ((r.dt - last[a]).days if a in last else np.nan)
                           - ((r.dt - last[b]).days if b in last else np.nan),
            "diff_age": ag(a) - ag(b),
            "both_exp": int(na > 0 and nb > 0),
            "dt": r.dt, "target": t, "key": r.key,
        })
        sa = 1.0 if t == 0 else (0.5 if t == 1 else 0.0)
        exp = 1.0 / (1.0 + 10 ** ((eb - ea) / 400))
        elo[a] = ea + ELO_K * (sa - exp); elo[b] = eb + ELO_K * ((1 - sa) - (1 - exp))
        opp_sum[a] += eb; opp_sum[b] += ea; n_[a] += 1; n_[b] += 1
        ko = any(k in r.method for k in ("ko", "tko", "rtd"))
        if t == 0:
            wins[a] += 1; losses[b] += 1
            if ko: kof[a] += 1; koa[b] += 1
        elif t == 2:
            wins[b] += 1; losses[a] += 1
            if ko: kof[b] += 1; koa[a] += 1
        else:
            draws[a] += 1; draws[b] += 1
        last[a] = last[b] = r.dt

    data = pd.DataFrame(rows)
    data = data[data["both_exp"] == 1].reset_index(drop=True)

    # --- A/B symmetrization: deterministically flip ~half the rows ---
    m = data["key"].map(flip).values
    diffcols = [c for c in data.columns if c.startswith("diff_")]
    data.loc[m, diffcols] = -data.loc[m, diffcols].values
    # target 0<->2 swap on flipped rows (1=draw stays)
    tgt = data["target"].values.copy()
    tgt[m & (data["target"] == 0)] = 2
    tgt[m & (data["target"] == 2)] = 0
    data["target"] = tgt
    print(f"modelable: {len(data):,}   class balance (a/draw/b): {np.round(np.bincount(data['target'],minlength=3)/len(data),3)}")

    # Temporal split by chronological quantile (robust to whatever date range
    # the corpus currently spans — the pull fills in modern boxers last).
    data = data.sort_values("dt").reset_index(drop=True)
    n = len(data)
    tr = data.iloc[: int(n * 0.70)]
    va = data.iloc[int(n * 0.70) : int(n * 0.85)]
    te = data.iloc[int(n * 0.85) :]
    print(f"split: train={len(tr):,} val={len(va):,} test={len(te):,} "
          f"(train ≤{tr['dt'].max().date()}, test ≥{te['dt'].min().date()})")

    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, log_loss
    params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
              "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 60,
              "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
              "lambda_l2": 1.0, "verbosity": -1, "seed": 42}
    dtr = lgb.Dataset(tr[diffcols], label=tr["target"])
    dva = lgb.Dataset(va[diffcols], label=va["target"], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
    base = np.bincount(tr["target"], minlength=3) / len(tr)
    for name, s in (("VAL", va), ("TEST", te)):
        if not len(s): continue
        p = model.predict(s[diffcols], num_iteration=model.best_iteration); y = s["target"].values
        ll = log_loss(y, p, labels=[0, 1, 2]); acc = accuracy_score(y, p.argmax(1))
        llb = log_loss(y, np.tile(base, (len(y), 1)), labels=[0, 1, 2])
        acc_elo = accuracy_score(y, np.where(s["diff_elo"] > 0, 0, 2))
        print(f"\n{name} (n={len(s):,}):  model log-loss {ll:.4f} acc {acc:.3f} | base LL {llb:.4f} (beats: {ll<llb}) | Elo-pick acc {acc_elo:.3f}")
    imp = pd.Series(model.feature_importance("gain"), index=diffcols).sort_values(ascending=False)
    print("\ntop features:", ", ".join(f"{k}={v:,.0f}" for k, v in imp.head(6).items()))


if __name__ == "__main__":
    main()
