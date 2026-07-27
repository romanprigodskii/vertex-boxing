"""Is real age worth crawling 140k fighter profiles for?

We only have a date of birth for ~10% of fighters, and the model currently
uses "days since first recorded bout" as a stand-in. Before spending a crawl
on the other 90%, measure the thing directly: take the bouts where BOTH men
have a known DOB, and compare the same model with and without real age on the
same rows. Whatever that difference is, it is the ceiling on what the crawl
can buy.

  ../scraper/venv/bin/python scripts/age_value.py
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "scraper"))
from src.db import get_connection  # noqa: E402

ELO_K, ELO_INIT = 32.0, 1500.0
BASE = ["d_elo", "d_bouts", "d_wr", "d_layoff", "n_a", "n_b", "d_career_days",
        "d_ko_rate", "d_sos"]
AGE = ["age_a", "age_b", "d_age"]


def load() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("""
        select e.date::date as dt, b.fighter_a_id as a, b.fighter_b_id as b,
               b.winner_id, b.is_draw, b.method::text as method,
               fa.dob as dob_a, fb.dob as dob_b
        from bout b join event e on e.id = b.event_id
        join fighter fa on fa.id = b.fighter_a_id
        join fighter fb on fb.id = b.fighter_b_id
        where b.status = 'completed' and e.date is not null
          and (b.winner_id is not null or b.is_draw)
        order by e.date, b.id
    """, conn)
    conn.close()
    df["dt"] = pd.to_datetime(df["dt"])
    for c in ("dob_a", "dob_b"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df[(df["dt"] >= "1950-01-01") & (df["dt"] <= pd.Timestamp.today())].reset_index(drop=True)


def symmetrize(df: pd.DataFrame) -> pd.DataFrame:
    keys = df["dt"].dt.strftime("%Y%m%d") + "|" + df["a"].astype(str) + "|" + df["b"].astype(str)
    flip = keys.map(lambda k: hashlib.blake2b(k.encode(), digest_size=4).digest()[-1] % 2 == 1).values
    out = df.copy()
    for x, y in (("a", "b"), ("dob_a", "dob_b")):
        out.loc[flip, [x, y]] = out.loc[flip, [y, x]].values
    return out


def replay(df: pd.DataFrame) -> pd.DataFrame:
    elo, seen, wins, last = defaultdict(lambda: ELO_INIT), defaultdict(int), defaultdict(int), {}
    ko, sos, first = defaultdict(int), defaultdict(float), {}
    rows = []
    for r in df.itertuples(index=False):
        a, b = r.a, r.b
        ea, eb, na, nb = elo[a], elo[b], seen[a], seen[b]
        age_a = (r.dt - r.dob_a).days / 365.25 if pd.notna(r.dob_a) else np.nan
        age_b = (r.dt - r.dob_b).days / 365.25 if pd.notna(r.dob_b) else np.nan
        rows.append((
            ea - eb, na - nb,
            (wins[a] / na if na else 0.5) - (wins[b] / nb if nb else 0.5),
            ((r.dt - last[a]).days if a in last else 400) - ((r.dt - last[b]).days if b in last else 400),
            na, nb,
            ((r.dt - first[a]).days if a in first else 0) - ((r.dt - first[b]).days if b in first else 0),
            (ko[a] / na if na else 0.0) - (ko[b] / nb if nb else 0.0),
            (sos[a] / na if na else ELO_INIT) - (sos[b] / nb if nb else ELO_INIT),
            age_a, age_b, (age_a - age_b) if (age_a == age_a and age_b == age_b) else np.nan,
        ))
        sa = 0.5 if r.is_draw else (1.0 if r.winner_id == a else 0.0)
        exp = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))
        elo[a] = ea + ELO_K * (sa - exp); elo[b] = eb + ELO_K * ((1 - sa) - (1 - exp))
        seen[a] += 1; seen[b] += 1
        wins[a] += sa >= 1.0; wins[b] += sa <= 0.0
        last[a] = last[b] = r.dt
        first.setdefault(a, r.dt); first.setdefault(b, r.dt)
        sos[a] += eb; sos[b] += ea
        if (r.method or "") in {"ko", "tko", "rtd"} and sa >= 1.0:
            ko[a] += 1
        elif (r.method or "") in {"ko", "tko", "rtd"} and sa <= 0.0:
            ko[b] += 1
    return pd.DataFrame(rows, columns=BASE + AGE)


def run(name, X, cols, y, tr, te):
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, log_loss
    params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.04,
              "num_leaves": 31, "min_data_in_leaf": 60, "feature_fraction": 0.9,
              "bagging_fraction": 0.9, "bagging_freq": 5, "lambda_l2": 2.0,
              "verbosity": -1, "seed": 42}
    idx = np.where(tr)[0]; cut = int(len(idx) * 0.85)
    dtr = lgb.Dataset(X.iloc[idx[:cut]][cols], label=y[idx[:cut]])
    dva = lgb.Dataset(X.iloc[idx[cut:]][cols], label=y[idx[cut:]], reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    p = m.predict(X.iloc[te][cols], num_iteration=m.best_iteration)
    ll = log_loss(y[te], np.clip(p, 1e-6, 1 - 1e-6))
    print(f"  {name:<22} log-loss {ll:.4f} · accuracy {accuracy_score(y[te], p > .5):.3f}")
    return -np.log(np.clip(np.where(y[te] == 1, p, 1 - p), 1e-9, 1)), m


def main() -> None:
    df = symmetrize(load())
    X = replay(df)
    y = (df["winner_id"] == df["a"]).astype(int).values
    both = df["dob_a"].notna() & df["dob_b"].notna() & (~df["is_draw"])
    print(f"боёв всего {len(df):,} · с известным возрастом ОБОИХ: {both.sum():,} "
          f"({both.mean()*100:.1f}%)")

    # эмпирическая кривая возраста
    age = X["age_a"][both.values]
    won = y[both.values]
    print("\nвозраст → доля побед:")
    for lo, hi in ((16, 22), (22, 26), (26, 30), (30, 34), (34, 38), (38, 60)):
        s = (age >= lo) & (age < hi)
        if s.sum() > 200:
            print(f"  {lo}-{hi} лет: n={int(s.sum()):>6,} · побед {won[s.values].mean()*100:5.1f}%")

    cutoff = df.loc[both, "dt"].quantile(0.85)
    tr = (both & (df["dt"] <= cutoff)).values
    te = (both & (df["dt"] > cutoff)).values
    print(f"\nобучение {tr.sum():,} · тест {te.sum():,} (одни и те же бои)")
    per_b, _ = run("без возраста", X, BASE, y, tr, te)
    per_a, m = run("+ настоящий возраст", X, BASE + AGE, y, tr, te)

    d = per_b - per_a
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nвыигрыш от возраста: {d.mean():+.4f} log-loss · 95% [{lo:+.4f}, {hi:+.4f}]")
    print("вывод:", "возраст значимо помогает — профили качать стоит" if lo > 0
          else ("значимо вредит" if hi < 0 else "разница неотличима от нуля — качать профили ради возраста НЕ стоит"))
    imp = pd.Series(m.feature_importance("gain"), index=BASE + AGE).sort_values(ascending=False)
    print("вклад:", " · ".join(f"{k}={v:,.0f}" for k, v in imp.head(6).items()))


if __name__ == "__main__":
    main()
