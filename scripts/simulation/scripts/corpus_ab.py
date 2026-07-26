"""Does the BoxRec layer actually buy predictive power? A/B on IDENTICAL bouts.

The tempting comparison — clean-corpus log-loss vs full-corpus log-loss — is
invalid: the two are measured on different bout populations, and the clean
corpus is the notable-fighter layer, which is inherently easier to predict
(bigger skill gaps, longer histories). A model can score better there while
being worse at everything.

So: one identity space (the database), one evaluation set, two rating
histories.

  CLEAN  ratings replayed over only those bouts that also exist in the
         Wikipedia record corpus — i.e. what we would know without BoxRec
  FULL   ratings replayed over every bout we have

Both models are trained on their own history and scored on the SAME held-out
bouts (the clean-corpus ones, since those are the only rows CLEAN can predict).
The difference in log-loss is the honest value of layer B.

  ../scraper/venv/bin/python scripts/corpus_ab.py
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRAPER = ROOT / "scripts" / "scraper"
sys.path.insert(0, str(SCRAPER))
from src.db import get_connection  # noqa: E402

WIKI = ROOT / "imports" / "staging" / "wikipedia_bouts.parquet"
ELO_K, ELO_INIT = 32.0, 1500.0
_PAREN = re.compile(r"\[.*?\]|\(.*?\)")


def norm(s) -> str | None:
    if not isinstance(s, str) or not s.strip():
        return None
    s = _PAREN.sub("", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return " ".join(s.split()) or None


def key(a: str, b: str, ym: str) -> str:
    return f"{min(a, b)}|{max(a, b)}|{ym}"


def load_clean_keys() -> set[str]:
    """(pair, year-month) of every bout in the clean Wikipedia corpus."""
    df = pd.read_parquet(WIKI)
    cols = {c.lower(): c for c in df.columns}
    a_col = cols.get("subject_name") or cols.get("fighter")
    b_col = cols.get("opponent_name") or cols.get("opponent")
    d_col = cols.get("date")
    out = set()
    for a, b, d in zip(df[a_col], df[b_col], pd.to_datetime(df[d_col], errors="coerce")):
        na, nb = norm(a), norm(b)
        if na and nb and pd.notna(d):
            out.add(key(na, nb, d.strftime("%Y-%m")))
    print(f"чистый корпус: {len(df):,} строк → {len(out):,} уникальных боёв")
    return out


def load_db() -> pd.DataFrame:
    conn = get_connection()
    q = """
        select b.id, e.date::date as dt,
               fa.id as a_id, fb.id as b_id,
               fa.name_en as a_name, fb.name_en as b_name,
               b.winner_id, b.is_draw
        from bout b
        join event e on e.id = b.event_id
        join fighter fa on fa.id = b.fighter_a_id
        join fighter fb on fb.id = b.fighter_b_id
        where b.status = 'completed' and e.date is not null
          and (b.winner_id is not null or b.is_draw)
        order by e.date
    """
    df = pd.read_sql(q, conn)
    conn.close()
    df["dt"] = pd.to_datetime(df["dt"])
    df = df[(df["dt"] >= "1900-01-01") & (df["dt"] <= pd.Timestamp.today())]
    print(f"база: {len(df):,} боёв с исходом ({df['dt'].min().date()} → {df['dt'].max().date()})")
    return df.reset_index(drop=True)


def replay(df: pd.DataFrame, use_mask: np.ndarray) -> pd.DataFrame:
    """Point-in-time features. Ratings update ONLY on bouts where use_mask is
    True, but features are snapshotted for every row — that is what lets the
    CLEAN model predict a bout it was not allowed to learn from."""
    elo: dict = defaultdict(lambda: ELO_INIT)
    seen: dict = defaultdict(int)
    wins: dict = defaultdict(int)
    last: dict = {}
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        a, b = r.a_id, r.b_id
        ea, eb = elo[a], elo[b]
        rows.append((
            ea - eb,
            seen[a] - seen[b],
            (wins[a] / seen[a] if seen[a] else 0.5) - (wins[b] / seen[b] if seen[b] else 0.5),
            ((r.dt - last[a]).days if a in last else 400) - ((r.dt - last[b]).days if b in last else 400),
            seen[a], seen[b],
        ))
        if not use_mask[i]:
            continue
        sa = 0.5 if r.is_draw else (1.0 if r.winner_id == a else 0.0)
        exp = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))
        elo[a] = ea + ELO_K * (sa - exp)
        elo[b] = eb + ELO_K * ((1 - sa) - (1 - exp))
        seen[a] += 1; seen[b] += 1
        wins[a] += sa >= 1.0; wins[b] += sa <= 0.0
        last[a] = last[b] = r.dt
    return pd.DataFrame(rows, columns=["d_elo", "d_bouts", "d_wr", "d_layoff", "n_a", "n_b"])


def target(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["is_draw"], 1, np.where(df["winner_id"] == df["a_id"], 0, 2))


def evaluate(name: str, feats: pd.DataFrame, y: np.ndarray, train: np.ndarray,
             test: np.ndarray) -> tuple[float, float]:
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, log_loss
    params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
              "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 60,
              "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
              "lambda_l2": 1.0, "verbosity": -1, "seed": 42}
    cut = int(train.sum() * 0.85)
    tr_idx = np.where(train)[0][:cut]
    va_idx = np.where(train)[0][cut:]
    dtr = lgb.Dataset(feats.iloc[tr_idx], label=y[tr_idx])
    dva = lgb.Dataset(feats.iloc[va_idx], label=y[va_idx], reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    p = m.predict(feats.iloc[test], num_iteration=m.best_iteration)
    ll = log_loss(y[test], p, labels=[0, 1, 2])
    acc = accuracy_score(y[test], p.argmax(1))
    print(f"  {name:<6} log-loss {ll:.4f} · accuracy {acc:.3f}")
    return ll, acc, -np.log(np.clip(p[np.arange(len(p)), y[test]], 1e-15, 1))


def main() -> None:
    clean_keys = load_clean_keys()
    df = load_db()

    na = df["a_name"].map(norm)
    nb = df["b_name"].map(norm)
    ym = df["dt"].dt.strftime("%Y-%m")
    in_clean = np.array([
        bool(x and z and key(x, z, m) in clean_keys)
        for x, z, m in zip(na, nb, ym)
    ])
    print(f"боёв базы, найденных в чистом корпусе: {in_clean.sum():,} "
          f"({in_clean.mean()*100:.1f}%)")

    y = target(df)
    # split on the CLEAN subset's own chronology: the database is dominated by
    # recent BoxRec rows, so a whole-corpus quantile leaves almost no clean bouts
    # in the test window and the comparison drowns in noise
    cutoff = df.loc[in_clean, "dt"].quantile(0.85)
    is_test = (df["dt"] > cutoff) & in_clean          # оцениваем ТОЛЬКО чистые бои
    is_train = df["dt"] <= cutoff
    print(f"порог {cutoff.date()} · обучение {is_train.sum():,} · "
          f"тест {is_test.sum():,} (одни и те же бои для обеих моделей)")

    print("\nCLEAN — рейтинги только по боям из чистого корпуса")
    f_clean = replay(df, in_clean)
    ll_c, acc_c, per_c = evaluate("CLEAN", f_clean, y, (is_train & in_clean).values, is_test.values)

    print("FULL — рейтинги по всем боям, включая BoxRec")
    f_full = replay(df, np.ones(len(df), dtype=bool))
    ll_f, acc_f, per_f = evaluate("FULL", f_full, y, is_train.values, is_test.values)

    # paired bootstrap on per-bout losses — the two models score the SAME rows,
    # so the pairing removes most of the variance and the CI is the honest answer
    d = per_c - per_f
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nразница log-loss (CLEAN − FULL): {d.mean():+.4f} "
          f"· 95% интервал [{lo:+.4f}, {hi:+.4f}] · n={len(d):,}")
    if lo > 0:
        print("вывод: слой B статистически значимо помогает")
    elif hi < 0:
        print("вывод: слой B статистически значимо ВРЕДИТ")
    else:
        print("вывод: разница неотличима от нуля — слой B не оправдан этим тестом")


if __name__ == "__main__":
    main()
