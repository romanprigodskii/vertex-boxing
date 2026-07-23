"""End-to-end leak-free prototype on the open-boxing title-bout corpus.

Proves the whole pipeline on data already on disk (no waiting on scrapes):
chronological replay → point-in-time Elo + record + strength-of-schedule
features (snapshot strictly BEFORE each bout) → 3-outcome LightGBM
(win_a / draw / win_b) → temporal split → honest metrics vs baselines.

Corpus is TITLE fights only (elite, the sharp end of the market), so this is a
pipeline + calibration sanity check, NOT the soft-market edge test — that needs
odds + the regional tail. Run: python scripts/prototype_openboxing.py
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BOUTS = ROOT / "imports" / "openboxing" / "bouts.csv"
CHAMPS = ROOT / "imports" / "openboxing" / "champions.csv"

ELO_K = 32.0
ELO_INIT = 1500.0


def norm_name(s) -> str | None:
    if not isinstance(s, str) or not s.strip():
        return None
    return " ".join(s.strip().lower().split())


def main() -> None:
    df = pd.read_csv(BOUTS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].sort_values("date").reset_index(drop=True)

    # --- target: 0=win_a, 1=draw, 2=win_b ---
    w = df["winner"].astype(str).str.upper()
    method = df["method_of_victory"].astype(str).str.lower()
    target = np.where(w.str.contains("BOXER A"), 0, np.where(w.str.contains("BOXER B"), 2, 1))
    # rows with no decisive winner AND not clearly a draw → drop
    is_draw = w.str.contains("DRAW") | method.str.contains("draw")
    keep = w.str.contains("BOXER A") | w.str.contains("BOXER B") | is_draw
    df, target = df[keep].reset_index(drop=True), target[keep.values]

    champs = pd.read_csv(CHAMPS)
    champs["born"] = pd.to_datetime(champs["born"], errors="coerce")
    dob = dict(zip(champs["champion_id"], champs["born"]))

    # --- chronological replay: state per fighter (by normalized name) ---
    elo: dict[str, float] = defaultdict(lambda: ELO_INIT)
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    draws: dict[str, int] = defaultdict(int)
    kos_for: dict[str, int] = defaultdict(int)
    kos_against: dict[str, int] = defaultdict(int)
    bouts_n: dict[str, int] = defaultdict(int)
    opp_elo_sum: dict[str, float] = defaultdict(float)
    last_date: dict[str, object] = {}

    rows: list[dict] = []
    for i, r in df.iterrows():
        a, b = norm_name(r["boxer_a_name"]), norm_name(r["boxer_b_name"])
        if not a or not b or a == b:
            continue
        t = int(target[i])
        ea, eb = elo[a], elo[b]

        def rate(d, n):
            return (d / n) if n else np.nan

        na, nb = bouts_n[a], bouts_n[b]
        # age from champion DOB when available
        def age(cid, when):
            d = dob.get(cid)
            return (when - d).days / 365.25 if pd.notna(d) else np.nan

        feat = {
            "diff_elo": ea - eb,
            "diff_winrate": rate(wins[a], na) - rate(wins[b], nb),
            "diff_ko_rate": rate(kos_for[a], na) - rate(kos_for[b], nb),
            "diff_koed_rate": rate(kos_against[a], na) - rate(kos_against[b], nb),
            "diff_bouts": na - nb,
            "diff_sos": (rate(opp_elo_sum[a], na) - rate(opp_elo_sum[b], nb)),
            "diff_layoff": (
                ((r["date"] - last_date[a]).days if a in last_date else np.nan)
                - ((r["date"] - last_date[b]).days if b in last_date else np.nan)
            ),
            "diff_age": age(r["boxer_a_champion_id"], r["date"]) - age(r["boxer_b_champion_id"], r["date"]),
            "scheduled_rounds": r["scheduled_rounds"],
            "both_experienced": int(na > 0 and nb > 0),
            "date": r["date"],
            "target": t,
        }
        rows.append(feat)

        # --- update state AFTER snapshot ---
        sa = 1.0 if t == 0 else (0.5 if t == 1 else 0.0)
        exp_a = 1.0 / (1.0 + 10 ** ((eb - ea) / 400))
        elo[a] = ea + ELO_K * (sa - exp_a)
        elo[b] = eb + ELO_K * ((1 - sa) - (1 - exp_a))
        opp_elo_sum[a] += eb
        opp_elo_sum[b] += ea
        bouts_n[a] += 1
        bouts_n[b] += 1
        ko = "ko" in method.iloc[i] or "tko" in method.iloc[i] or "rtd" in method.iloc[i]
        if t == 0:
            wins[a] += 1; losses[b] += 1
            if ko: kos_for[a] += 1; kos_against[b] += 1
        elif t == 2:
            wins[b] += 1; losses[a] += 1
            if ko: kos_for[b] += 1; kos_against[a] += 1
        else:
            draws[a] += 1; draws[b] += 1
        last_date[a] = last_date[b] = r["date"]

    data = pd.DataFrame(rows)
    data = data[data["both_experienced"] == 1].reset_index(drop=True)  # need history to predict
    print(f"modelable bouts (both have prior title-fight history): {len(data):,}")
    print("class balance (0=win_a,1=draw,2=win_b):", np.bincount(data["target"]) / len(data))

    # --- temporal split ---
    feat_cols = [c for c in data.columns if c.startswith(("diff_", "scheduled_"))]
    tr = data[data["date"] < "2010-01-01"]
    va = data[(data["date"] >= "2010-01-01") & (data["date"] < "2017-01-01")]
    te = data[data["date"] >= "2017-01-01"]
    print(f"split: train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, log_loss

    dtr = lgb.Dataset(tr[feat_cols], label=tr["target"])
    dva = lgb.Dataset(va[feat_cols], label=va["target"], reference=dtr)
    params = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 40,
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
        "lambda_l2": 1.0, "verbosity": -1, "seed": 42,
    }
    model = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80, verbose=False)])

    for name, s in (("VAL", va), ("TEST", te)):
        if len(s) == 0:
            continue
        p = model.predict(s[feat_cols], num_iteration=model.best_iteration)
        y = s["target"].values
        ll = log_loss(y, p, labels=[0, 1, 2])
        acc = accuracy_score(y, p.argmax(1))
        # baselines
        base_rate = np.bincount(tr["target"], minlength=3) / len(tr)
        ll_base = log_loss(y, np.tile(base_rate, (len(y), 1)), labels=[0, 1, 2])
        elo_pick = np.where(s["diff_elo"] > 0, 0, 2)
        acc_elo = accuracy_score(y, elo_pick)
        print(f"\n{name} (n={len(s):,}):")
        print(f"  model   log-loss {ll:.4f}   accuracy {acc:.3f}")
        print(f"  base-rate log-loss {ll_base:.4f}   (model beats base: {ll < ll_base})")
        print(f"  higher-Elo-picks accuracy {acc_elo:.3f}")

    imp = pd.Series(model.feature_importance(importance_type="gain"), index=feat_cols).sort_values(ascending=False)
    print("\ntop features by gain:")
    for k, v in imp.head(8).items():
        print(f"  {k:<20} {v:,.0f}")


if __name__ == "__main__":
    main()
