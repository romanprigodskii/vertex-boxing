"""Model against the closing line, on the bouts the market actually prices.

This is the only comparison that matters: beating a base rate is easy, beating
the price is the whole thesis. Both sides are scored on the SAME bouts, as a
two-way market (draws dropped — the odds file has no draw price), with the
bookmaker's margin divided out of the implied probabilities.

Two traps this avoids, both found the hard way:
  * A/B slot order. Our database lists the winner first 87% of the time, so a
    model that always answers A scores 0.87. Rows are symmetrized first.
  * Joining on names + month, order-sensitively. That silently keeps only the
    bouts where both sources happened to order the corners the same way, which
    correlates with who won. Match on the sorted pair + exact date, then align
    the two prices to our corners BY NAME.

  ../scraper/venv/bin/python scripts/market_vs_model.py
"""

from __future__ import annotations

import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "scraper"))
from src.db import get_connection  # noqa: E402

ODDS = ROOT / "imports" / "staging" / "proboxingodds.parquet"
ELO_K, ELO_INIT = 32.0, 1500.0
# division → its nominal pound limit; a real ordering beats a category code
DIV_LBS = {"minimumweight": 105, "light_flyweight": 108, "flyweight": 112,
           "super_flyweight": 115, "bantamweight": 118, "super_bantamweight": 122,
           "featherweight": 126, "super_featherweight": 130, "lightweight": 135,
           "super_lightweight": 140, "welterweight": 147, "super_welterweight": 154,
           "middleweight": 160, "super_middleweight": 168, "light_heavyweight": 175,
           "cruiserweight": 200, "heavyweight": 250}
_PAREN = re.compile(r"\[.*?\]|\(.*?\)")
FEATS = ["d_elo", "d_bouts", "d_wr", "d_layoff", "n_a", "n_b",
         "d_home", "d_promo_ties", "promo_bouts", "city_home_bias",
         # what the market obviously knows and we were not telling the model
         "sched_rounds",   # 4/6/8/10/12 — the single best proxy for the level of the bout
         "d_ko_rate",      # how often each man stops people
         "d_koed_rate",    # how often each man has BEEN stopped — the chin
         "d_sos",          # mean Elo of the opposition faced so far
         "d_losses",       # a padded record and a real one differ here
         "d_career_days",  # time since first recorded bout, as an age proxy
         # weight class and RECENT form — career averages hide a man in decline
         "weight_lbs",     # division as its nominal limit: heavies stop people, flies do not
         "d_form3",        # win rate over the last three bouts
         "d_sos3",         # mean opponent Elo over the last three
         "d_momentum"]     # Elo gained or lost over the last three


def norm(s):
    if not isinstance(s, str) or not s.strip():
        return None
    s = _PAREN.sub("", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return " ".join(s.split()) or None


def load_db() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("""
        select e.date::date as dt, b.fighter_a_id as a, b.fighter_b_id as b,
               fa.name_en as a_name, fb.name_en as b_name,
               b.winner_id, b.is_draw, b.method::text as method,
               coalesce(b.scheduled_rounds, 0) as sched, b.weight_class::text as div,
               e.location_country as country, e.location_city as city, e.promoter
        from bout b join event e on e.id = b.event_id
        join fighter fa on fa.id = b.fighter_a_id
        join fighter fb on fb.id = b.fighter_b_id
        where b.status = 'completed' and e.date is not null
          and (b.winner_id is not null or b.is_draw)
        order by e.date, b.id
    """, conn)
    conn.close()
    df["dt"] = pd.to_datetime(df["dt"])
    df = df[(df["dt"] >= "1950-01-01") & (df["dt"] <= pd.Timestamp.today())]
    return df.reset_index(drop=True)


def symmetrize(df: pd.DataFrame) -> pd.DataFrame:
    keys = df["dt"].dt.strftime("%Y%m%d") + "|" + df["a"].astype(str) + "|" + df["b"].astype(str)
    flip = keys.map(lambda k: hashlib.blake2b(k.encode(), digest_size=4).digest()[-1] % 2 == 1).values
    out = df.copy()
    for x, y in (("a", "b"), ("a_name", "b_name")):
        out.loc[flip, [x, y]] = out.loc[flip, [y, x]].values
    return out


def replay(df: pd.DataFrame) -> pd.DataFrame:
    elo, seen, wins, last = defaultdict(lambda: ELO_INIT), defaultdict(int), defaultdict(int), {}
    ctry = defaultdict(lambda: defaultdict(int))
    promo = defaultdict(lambda: defaultdict(int))
    promo_n, city_n, city_home = defaultdict(int), defaultdict(int), defaultdict(int)
    ko, koed, losses = defaultdict(int), defaultdict(int), defaultdict(int)
    sos_sum = defaultdict(float)
    first = {}
    from collections import deque
    form = defaultdict(lambda: deque(maxlen=3))      # last three results
    opp3 = defaultdict(lambda: deque(maxlen=3))      # last three opponents' Elo
    elo3 = defaultdict(lambda: deque(maxlen=3))      # Elo before each of the last three
    STOP = {"ko", "tko", "rtd"}
    rows = []
    for r in df.itertuples(index=False):
        a, b = r.a, r.b
        ea, eb, na, nb = elo[a], elo[b], seen[a], seen[b]
        ha = (ctry[a][r.country] / na) if (na and r.country) else 0.0
        hb = (ctry[b][r.country] / nb) if (nb and r.country) else 0.0
        rows.append((
            ea - eb, na - nb,
            (wins[a] / na if na else 0.5) - (wins[b] / nb if nb else 0.5),
            ((r.dt - last[a]).days if a in last else 400) - ((r.dt - last[b]).days if b in last else 400),
            na, nb, ha - hb,
            (promo[a][r.promoter] - promo[b][r.promoter]) if r.promoter else 0,
            promo_n[r.promoter] if r.promoter else 0,
            (city_home[r.city] / city_n[r.city]) if (r.city and city_n[r.city] >= 20) else np.nan,
            r.sched or np.nan,
            (ko[a] / na if na else 0.0) - (ko[b] / nb if nb else 0.0),
            (koed[a] / na if na else 0.0) - (koed[b] / nb if nb else 0.0),
            (sos_sum[a] / na if na else ELO_INIT) - (sos_sum[b] / nb if nb else ELO_INIT),
            losses[a] - losses[b],
            ((r.dt - first[a]).days if a in first else 0) - ((r.dt - first[b]).days if b in first else 0),
            DIV_LBS.get(r.div, np.nan),
            (np.mean(form[a]) if form[a] else 0.5) - (np.mean(form[b]) if form[b] else 0.5),
            (np.mean(opp3[a]) if opp3[a] else ELO_INIT) - (np.mean(opp3[b]) if opp3[b] else ELO_INIT),
            ((ea - elo3[a][0]) if elo3[a] else 0.0) - ((eb - elo3[b][0]) if elo3[b] else 0.0),
        ))
        sa = 0.5 if r.is_draw else (1.0 if r.winner_id == a else 0.0)
        exp = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))
        elo[a] = ea + ELO_K * (sa - exp)
        elo[b] = eb + ELO_K * ((1 - sa) - (1 - exp))
        seen[a] += 1; seen[b] += 1
        wins[a] += sa >= 1.0; wins[b] += sa <= 0.0
        last[a] = last[b] = r.dt
        first.setdefault(a, r.dt); first.setdefault(b, r.dt)
        form[a].append(sa); form[b].append(1 - sa)
        opp3[a].append(eb); opp3[b].append(ea)
        elo3[a].append(ea); elo3[b].append(eb)
        sos_sum[a] += eb; sos_sum[b] += ea
        if sa <= 0.0: losses[a] += 1
        if sa >= 1.0: losses[b] += 1
        if (r.method or "") in STOP:
            win, lose = (a, b) if sa >= 1.0 else ((b, a) if sa <= 0.0 else (None, None))
            if win:
                ko[win] += 1; koed[lose] += 1
        if r.country:
            ctry[a][r.country] += 1; ctry[b][r.country] += 1
        if r.promoter:
            promo[a][r.promoter] += 1; promo[b][r.promoter] += 1; promo_n[r.promoter] += 1
        if r.city:
            city_n[r.city] += 1
            city_home[r.city] += (sa >= 1.0) if ha >= hb else (sa <= 0.0)
    return pd.DataFrame(rows, columns=FEATS)


def main() -> None:
    df = symmetrize(load_db())
    feats = replay(df)
    df["na"], df["nb"] = df["a_name"].map(norm), df["b_name"].map(norm)
    df["key"] = [f"{min(x, y)}|{max(x, y)}|{d:%Y-%m-%d}" if x and y else None
                 for x, y, d in zip(df["na"], df["nb"], df["dt"])]

    od = pd.read_parquet(ODDS)
    od["dt"] = pd.to_datetime(od["date"])
    od["na"], od["nb"] = od["a"].map(norm), od["b"].map(norm)
    od = od.dropna(subset=["na", "nb", "close_a", "close_b"])
    od["key"] = [f"{min(x, y)}|{max(x, y)}|{d:%Y-%m-%d}"
                 for x, y, d in zip(od["na"], od["nb"], od["dt"])]
    od = od.drop_duplicates("key")[["key", "na", "close_a", "close_b"]]

    j = df.reset_index().merge(od, on="key", how="inner", suffixes=("", "_o"))
    j = j[~j["is_draw"]]
    # align the two prices to OUR corners by name, never by position
    same = (j["na"] == j["na_o"]).values
    ca = np.where(same, j["close_a"], j["close_b"]).astype(float)
    cb = np.where(same, j["close_b"], j["close_a"]).astype(float)
    p_mkt = (1 / ca) / ((1 / ca) + (1 / cb))          # margin divided out
    y = (j["winner_id"] == j["a"]).astype(int).values  # 1 = our corner A won
    idx = j["index"].values
    print(f"боёв с котировкой и исходом: {len(j):,} "
          f"({j['dt'].min().date()} → {j['dt'].max().date()})")

    cutoff = pd.Series(j["dt"]).quantile(0.6)
    tr, te = (j["dt"] <= cutoff).values, (j["dt"] > cutoff).values
    print(f"порог {cutoff.date()} · обучение {tr.sum():,} · тест {te.sum():,}")

    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, log_loss
    params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03,
              "num_leaves": 15, "min_data_in_leaf": 40, "feature_fraction": 0.9,
              "bagging_fraction": 0.9, "bagging_freq": 5, "lambda_l2": 5.0,
              "verbosity": -1, "seed": 42}
    X = feats.iloc[idx].reset_index(drop=True)
    # Train on the WHOLE corpus up to the cutoff, not on the 470 bouts that
    # happen to carry odds — the market is priced by people who watch the whole
    # sport, and starving the model of 99.8% of its history is not a fair test.
    y_all = np.where(df["is_draw"], 0, (df["winner_id"] == df["a"]).astype(int))
    keep = (~df["is_draw"]).values & (df["dt"] <= cutoff).values
    big = np.where(keep)[0]
    cut = int(len(big) * 0.9)
    dtr = lgb.Dataset(feats.iloc[big[:cut]], label=y_all[big[:cut]])
    dva = lgb.Dataset(feats.iloc[big[cut:]], label=y_all[big[cut:]], reference=dtr)
    mdl = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(80, verbose=False)])
    print(f"модель обучена на {len(big):,} боях (было 470)")
    p_mod = mdl.predict(X.iloc[te], num_iteration=mdl.best_iteration)
    pm, yy = p_mkt[te], y[te]

    ll_mod = log_loss(yy, np.clip(p_mod, 1e-6, 1 - 1e-6))
    ll_mkt = log_loss(yy, np.clip(pm, 1e-6, 1 - 1e-6))
    print(f"\n  МОДЕЛЬ  log-loss {ll_mod:.4f} · accuracy {accuracy_score(yy, p_mod > .5):.3f}")
    print(f"  РЫНОК   log-loss {ll_mkt:.4f} · accuracy {accuracy_score(yy, pm > .5):.3f}")

    d = (-np.log(np.clip(np.where(yy == 1, pm, 1 - pm), 1e-9, 1))
         - -np.log(np.clip(np.where(yy == 1, p_mod, 1 - p_mod), 1e-9, 1)))
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  модель − рынок: {d.mean():+.4f} log-loss · 95% [{lo:+.4f}, {hi:+.4f}]")
    print("  вывод:", "модель ЛУЧШЕ рынка" if lo > 0 else
          ("рынок лучше модели" if hi < 0 else "разница неотличима от нуля"))

    comp = (pm > 0.3) & (pm < 0.7)
    if comp.sum() > 50:
        llm = log_loss(yy[comp], np.clip(p_mod[comp], 1e-6, 1 - 1e-6))
        llk = log_loss(yy[comp], np.clip(pm[comp], 1e-6, 1 - 1e-6))
        print(f"\n  конкурентные бои (цена 30-70%), n={comp.sum():,}: "
              f"модель {llm:.4f} · рынок {llk:.4f}")


if __name__ == "__main__":
    main()
