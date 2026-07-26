"""Do venue, referee, judges and promoter add anything a rating cannot see?

This is the part of the thesis that the clean corpus cannot test at all: soft
regional markets are supposed to misprice exactly the things a bare rating
ignores — a fighter brought to the promoter's own town, a referee quick to
wave it off, judges who reliably find for the house.

Every added feature is point-in-time, built in one chronological pass and
snapshotted strictly BEFORE the bout it describes:

  home        share of each fighter's PREVIOUS bouts held in this country
              (we have no nationality field; where a man has been fighting is
              the better proxy anyway — it is about the crowd and the judges)
  promoter    how many previous bouts each man had on THIS promoter's cards —
              the house-fighter signal, and the one I expect to matter
  referee     that referee's prior rate of stopping fights inside the distance
  judges      those judges' prior rate of scoring for the home-side fighter
  weight      weigh-in difference in lb

Both models are scored on the SAME rows (bouts that actually carry the fields),
so the only difference is whether the extra columns are visible.

  ../scraper/venv/bin/python scripts/home_bias.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "scraper"))
from src.db import get_connection  # noqa: E402

ELO_K, ELO_INIT = 32.0, 1500.0
STOPPAGE = {"ko", "tko", "rtd"}
BASE = ["d_elo", "d_bouts", "d_wr", "d_layoff", "n_a", "n_b"]
EXTRA = ["d_home", "d_promo_ties", "ref_stop_rate", "ref_bouts",
         "judge_home_bias", "d_weight", "promo_bouts"]


def load() -> pd.DataFrame:
    conn = get_connection()
    q = """
        select e.date::date as dt, b.fighter_a_id as a, b.fighter_b_id as b,
               b.winner_id, b.is_draw, b.method::text as method,
               e.location_country as country, e.promoter, b.referee,
               b.judges, b.a_weight_lbs, b.b_weight_lbs
        from bout b join event e on e.id = b.event_id
        where b.status = 'completed' and e.date is not null
          and (b.winner_id is not null or b.is_draw)
        order by e.date, b.id
    """
    df = pd.read_sql(q, conn)
    conn.close()
    df["dt"] = pd.to_datetime(df["dt"])
    df = df[(df["dt"] >= "1950-01-01") & (df["dt"] <= pd.Timestamp.today())]
    print(f"боёв с исходом: {len(df):,} ({df['dt'].min().date()} → {df['dt'].max().date()})")
    return df.reset_index(drop=True)


def judge_ids(v) -> list[str]:
    if not v:
        return []
    try:
        arr = v if isinstance(v, list) else json.loads(v)
        return [str(j.get("boxrec_id")) for j in arr if j.get("boxrec_id")]
    except Exception:  # noqa: BLE001
        return []


def replay(df: pd.DataFrame) -> pd.DataFrame:
    elo: dict = defaultdict(lambda: ELO_INIT)
    seen: dict = defaultdict(int)
    wins: dict = defaultdict(int)
    last: dict = {}
    ctry: dict = defaultdict(lambda: defaultdict(int))     # fighter → country → n
    promo: dict = defaultdict(lambda: defaultdict(int))    # fighter → promoter → n
    ref_n: dict = defaultdict(int)
    ref_stop: dict = defaultdict(int)
    jud_n: dict = defaultdict(int)
    jud_home: dict = defaultdict(int)
    promo_n: dict = defaultdict(int)

    rows = []
    for r in df.itertuples(index=False):
        a, b = r.a, r.b
        ea, eb = elo[a], elo[b]
        na, nb = seen[a], seen[b]

        # --- home: where each man has been fighting, before today
        ha = (ctry[a][r.country] / na) if (na and r.country) else 0.0
        hb = (ctry[b][r.country] / nb) if (nb and r.country) else 0.0
        # --- promoter ties: bouts already fought on this promoter's cards
        pa = promo[a][r.promoter] if r.promoter else 0
        pb = promo[b][r.promoter] if r.promoter else 0

        jids = judge_ids(r.judges)
        jb = [jud_home[j] / jud_n[j] for j in jids if jud_n[j] >= 5]
        rows.append((
            ea - eb, na - nb,
            (wins[a] / na if na else 0.5) - (wins[b] / nb if nb else 0.5),
            ((r.dt - last[a]).days if a in last else 400) - ((r.dt - last[b]).days if b in last else 400),
            na, nb,
            ha - hb, pa - pb,
            (ref_stop[r.referee] / ref_n[r.referee]) if (r.referee and ref_n[r.referee] >= 5) else np.nan,
            ref_n[r.referee] if r.referee else 0,
            float(np.mean(jb)) if jb else np.nan,
            (float(r.a_weight_lbs) - float(r.b_weight_lbs))
            if (r.a_weight_lbs is not None and r.b_weight_lbs is not None) else np.nan,
            promo_n[r.promoter] if r.promoter else 0,
        ))

        # ---- update state AFTER snapshotting
        sa = 0.5 if r.is_draw else (1.0 if r.winner_id == a else 0.0)
        exp = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))
        elo[a] = ea + ELO_K * (sa - exp)
        elo[b] = eb + ELO_K * ((1 - sa) - (1 - exp))
        seen[a] += 1; seen[b] += 1
        wins[a] += sa >= 1.0; wins[b] += sa <= 0.0
        last[a] = last[b] = r.dt
        if r.country:
            ctry[a][r.country] += 1; ctry[b][r.country] += 1
        if r.promoter:
            promo[a][r.promoter] += 1; promo[b][r.promoter] += 1
            promo_n[r.promoter] += 1
        if r.referee:
            ref_n[r.referee] += 1
            ref_stop[r.referee] += (r.method or "") in STOPPAGE
        if jids:
            home_side_a = ha >= hb          # who the crowd belonged to
            for j in jids:
                jud_n[j] += 1
                jud_home[j] += (sa >= 1.0) if home_side_a else (sa <= 0.0)
    return pd.DataFrame(rows, columns=BASE + EXTRA)


def run(name: str, feats: pd.DataFrame, cols: list[str], y: np.ndarray,
        tr: np.ndarray, te: np.ndarray):
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, log_loss
    params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
              "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 60,
              "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
              "lambda_l2": 1.0, "verbosity": -1, "seed": 42}
    idx = np.where(tr)[0]
    cut = int(len(idx) * 0.85)
    dtr = lgb.Dataset(feats.iloc[idx[:cut]][cols], label=y[idx[:cut]])
    dva = lgb.Dataset(feats.iloc[idx[cut:]][cols], label=y[idx[cut:]], reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    p = m.predict(feats.iloc[te][cols], num_iteration=m.best_iteration)
    ll = log_loss(y[te], p, labels=[0, 1, 2])
    acc = accuracy_score(y[te], p.argmax(1))
    print(f"  {name:<18} log-loss {ll:.4f} · accuracy {acc:.3f}")
    per = -np.log(np.clip(p[np.arange(len(p)), y[te]], 1e-15, 1))
    return ll, per, m, cols


def main() -> None:
    df = load()
    feats = replay(df)
    y = np.where(df["is_draw"], 1, np.where(df["winner_id"] == df["a"], 0, 2))

    # only rows that actually carry the new fields — both models see the same rows
    has = df["referee"].notna() & df["country"].notna() & df["promoter"].notna()
    cutoff = df.loc[has, "dt"].quantile(0.85)
    te = ((df["dt"] > cutoff) & has).values
    tr = ((df["dt"] <= cutoff) & has).values
    print(f"строк с новыми полями: {has.sum():,} · порог {cutoff.date()} · "
          f"обучение {tr.sum():,} · тест {te.sum():,}")

    print("\nбез новых полей / с новыми полями (одни и те же бои)")
    ll_b, per_b, *_ = run("только рейтинги", feats, BASE, y, tr, te)
    ll_x, per_x, m, cols = run("+ дом/рефери/судьи", feats, BASE + EXTRA, y, tr, te)

    d = per_b - per_x
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nвыигрыш от новых полей: {d.mean():+.4f} log-loss · "
          f"95% интервал [{lo:+.4f}, {hi:+.4f}]")
    print("вывод:", "новые поля значимо помогают" if lo > 0 else
          ("значимо вредят" if hi < 0 else "разница неотличима от нуля"))

    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("\nвклад признаков:", " · ".join(f"{k}={v:,.0f}" for k, v in imp.head(8).items()))


if __name__ == "__main__":
    main()
