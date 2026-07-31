"""The one scoreboard: model against the closing line, on the same bouts.

Everything else is a knob on this. Feature sets, calibration population,
training weights and the market blend are flags, so two runs differ by exactly
what the flag says and by nothing else — the replay is cached per corpus tag.

  ./venv/bin/python scripts/market_eval.py --tag post-ingest --feats all
  ./venv/bin/python scripts/market_eval.py --tag post-ingest --feats base --label baseline
  ./venv/bin/python scripts/market_eval.py --tag post-ingest --calib quoted --weight quoted --blend
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
from src import features as F  # noqa: E402

ODDS = ROOT / "imports" / "staging" / "proboxingodds.parquet"
CACHE = ROOT / "imports" / "staging"
_PAREN = re.compile(r"\[.*?\]|\(.*?\)")

SETS = {
    "base": F.BASE,
    "base+record": F.BASE + F.RECORD,
    "base+sos2": F.BASE + F.SOS2,
    "base+glicko": F.BASE + F.GLICKO,
    "base+age": F.BASE + F.AGE,
    "base+level": F.BASE + F.LEVEL,
    "all": F.ALL,
    "all-noage": F.BASE + F.RECORD + F.SOS2 + F.GLICKO + F.LEVEL,
    # Recursive strength-of-schedule looked harmful on the 416 quoted bouts
    # (-0.0135 [-0.0253, -0.0017]) and helpful on the 75,779-bout corpus
    # holdout (+0.0023 [+0.0015, +0.0031]). The second instrument is the one
    # with the power, so it stays in — this set is kept only to show the
    # difference the choice makes.
    "best": F.BASE + F.RECORD + F.GLICKO + F.AGE + F.LEVEL,
}


def norm(s):
    """Accent-folding: 'Álvarez' and 'Alvarez' are the same man, and the version
    without NFKD silently dropped every Spanish-named fighter from the join."""
    if not isinstance(s, str) or not s.strip():
        return None
    s = _PAREN.sub("", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()) or None


def arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def build(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Symmetrized corpus + its features, cached — the replay is minutes."""
    fc, dc = CACHE / f"feats_{tag}.parquet", CACHE / f"sym_{tag}.parquet"
    if fc.exists() and dc.exists():
        return pd.read_parquet(dc), pd.read_parquet(fc)
    df = F.symmetrize(F.load(tag))
    print(f"replaying {len(df):,} bouts…", flush=True)
    feats = F.replay(df)
    df.to_parquet(dc, index=False)
    feats.to_parquet(fc, index=False)
    return df, feats


def join_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Pair + date, one day either side. The pair has to be exact; the day is
    allowed to slip because a card that starts late local time is dated the next
    day by one source and not the other."""
    df = df.copy()
    df["na"], df["nb"] = df["a_name"].map(norm), df["b_name"].map(norm)
    df["pair"] = [f"{min(x, y)}|{max(x, y)}" if x and y else None
                  for x, y in zip(df["na"], df["nb"])]

    od = pd.read_parquet(ODDS)
    od["dt"] = pd.to_datetime(od["date"])
    od["na"], od["nb"] = od["a"].map(norm), od["b"].map(norm)
    od = od.dropna(subset=["na", "nb", "close_a", "close_b"])
    od["pair"] = [f"{min(x, y)}|{max(x, y)}" for x, y in zip(od["na"], od["nb"])]
    od = od.drop_duplicates(subset=["pair", "dt"])

    dbi = df.reset_index()[["index", "pair", "dt"]].dropna(subset=["pair"])
    cand = dbi.merge(od[["pair", "dt", "na", "close_a", "close_b"]], on="pair",
                     how="inner", suffixes=("", "_o"))
    cand["gap"] = (cand["dt"] - cand["dt_o"]).abs().dt.days
    cand = cand[cand["gap"] <= 1].sort_values("gap")
    cand = cand.drop_duplicates("index").drop_duplicates(subset=["pair", "dt_o"])
    cand = cand.rename(columns={"na": "na_o"})
    j = df.reset_index().merge(cand[["index", "na_o", "close_a", "close_b", "gap"]],
                               on="index", how="inner")
    return j[~j["is_draw"]].reset_index(drop=True)


def bootstrap(d: np.ndarray, n: int = 4000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return tuple(np.percentile(boot, [2.5, 97.5]))


def main() -> None:  # noqa: PLR0915
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import accuracy_score, log_loss

    tag = arg("--tag", "post-ingest")
    fset = arg("--feats", "all")
    calib = arg("--calib", "all")        # all | quoted | matched
    weight = arg("--weight", "none")     # none | quoted
    label = arg("--label", fset)
    cols = SETS[fset]
    # leave-one-group-out: the only honest way to say which group carries the
    # gain, because groups overlap in what they explain
    drop = arg("--drop", "")
    if drop:
        gone = {"glicko": F.GLICKO, "record": F.RECORD, "sos2": F.SOS2,
                "age": F.AGE, "level": F.LEVEL, "base": F.BASE}[drop]
        cols = [c for c in cols if c not in gone]

    df, feats = build(tag)
    j = join_odds(df)
    print(f"[{label}] priced bouts with an outcome: {len(j):,} "
          f"({j['dt'].min().date()} → {j['dt'].max().date()}) · "
          f"same-day {(j['gap'] == 0).mean():.0%}")

    same = (j["na"] == j["na_o"]).values
    ca = np.where(same, j["close_a"], j["close_b"]).astype(float)
    cb = np.where(same, j["close_b"], j["close_a"]).astype(float)
    p_mkt = (1 / ca) / ((1 / ca) + (1 / cb))          # margin divided out
    y = (j["winner_id"].astype(str) == j["a"].astype(str)).astype(int).values
    idx = j["index"].values

    cutoff = pd.Series(j["dt"]).quantile(0.6)
    tr, te = (j["dt"] <= cutoff).values, (j["dt"] > cutoff).values
    print(f"  cutoff {cutoff.date()} · quoted train {tr.sum():,} · test {te.sum():,}")

    y_all = F.label(df)
    pre = (df["dt"] <= cutoff).values
    keep = (~df["is_draw"]).values & pre
    big = np.where(keep)[0]
    y_big = y_all[big].astype(float)
    w_big = np.ones(len(big))
    X = feats[cols]

    if "--draws" in sys.argv:
        # A draw used to be dropped, so the model was never shown a matchup so
        # even it ended level — and those are exactly the 30-70% bouts where it
        # is weakest. Each draw enters twice at half weight, once for each
        # corner: the target becomes "probability A does not lose", which is
        # what a two-way price without a draw line is really quoting.
        d_idx = np.where(df["is_draw"].values & pre)[0]
        big = np.concatenate([big, d_idx, d_idx])
        y_big = np.concatenate([y_big, np.ones(len(d_idx)), np.zeros(len(d_idx))])
        w_big = np.concatenate([w_big, np.full(len(d_idx), 0.5), np.full(len(d_idx), 0.5)])
        order = np.argsort(df["dt"].values[big], kind="stable")
        big, y_big, w_big = big[order], y_big[order], w_big[order]
        print(f"  draws folded in: {len(d_idx):,} of {len(big):,} training rows")

    # Train on the whole sport up to the cutoff: the price is set by people who
    # watch all of it, and starving the model of 99% of its history is not a
    # fair test. Optionally lean the sample towards the population that
    # actually gets quoted.
    if weight == "quoted":
        s = feats["sched_rounds"].to_numpy()
        n_min = np.minimum(feats["n_a"].to_numpy(), feats["n_b"].to_numpy())
        w_all = np.where((np.nan_to_num(s, nan=0) >= 8) & (n_min >= 8), 3.0, 1.0)
        w_big = w_big * w_all[big]
    w = w_big
    cut = int(len(big) * 0.9)
    dtr = lgb.Dataset(X.iloc[big[:cut]], label=y_big[:cut], weight=w[:cut])
    dva = lgb.Dataset(X.iloc[big[cut:]], label=y_big[cut:], weight=w[cut:],
                      reference=dtr)
    params = {"objective": "binary", "metric": "binary_logloss",
              "learning_rate": float(arg("--lr", "0.03")),
              "num_leaves": int(arg("--leaves", "31")),
              "min_data_in_leaf": int(arg("--minleaf", "40")),
              "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
              "lambda_l2": float(arg("--l2", "5.0")),
              "verbosity": -1, "seed": 42}
    mdl = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(150, verbose=False)])
    print(f"  trained on {len(big):,} bouts · {len(cols)} features · "
          f"{mdl.best_iteration} trees")

    # An ensemble averaged in LOGIT space. Three models that are wrong in
    # different places: gradient boosting on leaves, gradient boosting on
    # ordered target statistics, and a plain linear model that cannot overfit
    # a rare interaction the way a tree can.
    members = [("lgbm", lambda Z: mdl.predict(Z, num_iteration=mdl.best_iteration))]
    if "--ensemble" in sys.argv:
        from catboost import CatBoostClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6,
                                l2_leaf_reg=6.0, loss_function="Logloss",
                                random_seed=42, verbose=False,
                                early_stopping_rounds=150)
        cb.fit(X.iloc[big[:cut]], y_big[:cut], sample_weight=w[:cut],
               eval_set=(X.iloc[big[cut:]], y_big[cut:]))
        lin = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=0.1, max_iter=3000))
        lin.fit(X.iloc[big[:cut]], y_big[:cut],
                logisticregression__sample_weight=w[:cut])
        members += [("catboost", lambda Z: cb.predict_proba(Z)[:, 1]),
                    ("logreg", lambda Z: lin.predict_proba(Z)[:, 1])]
        print(f"  ensemble: {', '.join(n for n, _ in members)} "
              f"(catboost {cb.get_best_iteration()} trees)")

    def predict(Z):
        ps = [np.clip(fn(Z), 1e-6, 1 - 1e-6) for _, fn in members]
        lg = np.mean([np.log(p / (1 - p)) for p in ps], axis=0)
        return 1.0 / (1.0 + np.exp(-lg))

    # Calibration. Boosting on an imbalanced target is over-confident and
    # log-loss punishes that harder than being wrong; but a calibrator fitted on
    # the four-round regional tail is being asked about a population it never
    # saw, so the population it is fitted on is a knob.
    if calib == "quoted":
        src_X, src_y = X.iloc[idx[tr]], y[tr]
    elif calib == "matched":
        s = feats["sched_rounds"].to_numpy()[big[cut:]]
        n_min = np.minimum(feats["n_a"].to_numpy(), feats["n_b"].to_numpy())[big[cut:]]
        m = (np.nan_to_num(s, nan=0) >= 8) & (n_min >= 8)
        src_X, src_y = X.iloc[big[cut:][m]], y_big[cut:][m]
    else:
        src_X, src_y = X.iloc[big[cut:]], y_big[cut:]
    p_src = predict(src_X)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_src, src_y)
    raw_te = predict(X.iloc[idx[te]])
    p_mod = np.clip(iso.predict(raw_te), 1e-4, 1 - 1e-4)
    print(f"  calibration on {calib} (n={len(src_y):,}): "
          f"{log_loss(y[te], np.clip(raw_te, 1e-6, 1-1e-6)):.4f} → {log_loss(y[te], p_mod):.4f}")

    # The quoted test set is ~400 bouts, so it cannot see a gain of 0.005 —
    # every feature group came back "indistinguishable from zero" on it. The
    # whole post-cutoff corpus can: same model, same cutoff, 100x the bouts.
    # It says nothing about the market, and that is fine — it is the instrument
    # for "does this feature help", while the quoted set answers "do we beat
    # the price".
    post = (~df["is_draw"]).values & (df["dt"] > cutoff).values
    pidx = np.where(post)[0]
    p_corp = np.clip(iso.predict(predict(X.iloc[pidx])), 1e-4, 1 - 1e-4)
    y_corp = y_all[pidx]
    ll_corp = log_loss(y_corp, p_corp)
    print(f"  корпусный холдаут n={len(pidx):,}: log-loss {ll_corp:.4f} · "
          f"accuracy {accuracy_score(y_corp, p_corp > .5):.3f}")

    pm, yy = p_mkt[te], y[te]
    ll_mod, ll_mkt = log_loss(yy, p_mod), log_loss(yy, np.clip(pm, 1e-6, 1 - 1e-6))
    print(f"\n  МОДЕЛЬ  log-loss {ll_mod:.4f} · accuracy {accuracy_score(yy, p_mod > .5):.3f}")
    print(f"  РЫНОК   log-loss {ll_mkt:.4f} · accuracy {accuracy_score(yy, pm > .5):.3f}")
    d = (-np.log(np.clip(np.where(yy == 1, pm, 1 - pm), 1e-9, 1))
         - -np.log(np.clip(np.where(yy == 1, p_mod, 1 - p_mod), 1e-9, 1)))
    lo, hi = bootstrap(d)
    print(f"  модель − рынок: {d.mean():+.4f} · 95% [{lo:+.4f}, {hi:+.4f}]")

    comp = (pm > 0.3) & (pm < 0.7)
    if comp.sum() > 30:
        print(f"  конкурентные 30-70% (n={comp.sum():,}): "
              f"модель {log_loss(yy[comp], p_mod[comp]):.4f} · "
              f"рынок {log_loss(yy[comp], np.clip(pm[comp], 1e-6, 1-1e-6)):.4f}")

    out = {"label": label, "tag": tag, "feats": fset, "calib": calib,
           "weight": weight, "n_test": int(te.sum()), "n_train_quoted": int(tr.sum()),
           "ll_model": float(ll_mod), "ll_market": float(ll_mkt),
           "gap": float(d.mean()), "ci": [float(lo), float(hi)],
           "ll_corpus": float(ll_corp), "n_corpus_test": int(len(pidx)),
           "n_corpus": int(len(df)), "trees": int(mdl.best_iteration)}

    if "--blend" in sys.argv:
        # Does the model add anything TO the price? This is the only question
        # the data can actually answer: λ>0 means residual information even
        # while losing on raw log-loss, λ≈0 closes the thesis honestly.
        # λ CANNOT be fitted on the quoted train bouts against the main model:
        # those bouts are inside its training corpus, so its predictions there
        # are in-sample and λ comes out at 0.85 purely because the model
        # remembers them. Fit λ against an AUXILIARY model trained on an
        # earlier cutoff, for which the same bouts are genuinely unseen.
        c1 = pd.Series(j["dt"]).quantile(0.35)
        akeep = (~df["is_draw"]).values & (df["dt"] <= c1).values
        abig = np.where(akeep)[0]
        acut = int(len(abig) * 0.9)
        adtr = lgb.Dataset(X.iloc[abig[:acut]], label=y_all[abig[:acut]])
        adva = lgb.Dataset(X.iloc[abig[acut:]], label=y_all[abig[acut:]], reference=adtr)
        amdl = lgb.train(params, adtr, num_boost_round=4000, valid_sets=[adva],
                         callbacks=[lgb.early_stopping(150, verbose=False)])
        aiso = IsotonicRegression(out_of_bounds="clip").fit(
            amdl.predict(X.iloc[abig[acut:]], num_iteration=amdl.best_iteration),
            y_all[abig[acut:]])
        mid = ((j["dt"] > c1) & (j["dt"] <= cutoff)).values
        print(f"\n  λ подбирается на {mid.sum():,} боях после {c1.date()}, "
              f"вне обучения вспомогательной модели ({len(abig):,} боёв)")
        tr = mid                      # the fitting slice for the blend
        p_tr = np.clip(aiso.predict(amdl.predict(X.iloc[idx[tr]],
                                                 num_iteration=amdl.best_iteration)),
                       1e-4, 1 - 1e-4)
        def lg(p): return np.log(p / (1 - p))
        # ONE parameter, not two. A free two-weight logistic on 2,500 bouts
        # fits the scale of each logit as well as the mix and comes out with
        # weights like (+0.91, +0.40) that do not survive the test set. λ is
        # the question we actually asked: how much of the model does the price
        # not already contain.
        ltr_m, ltr_k = lg(p_tr), lg(np.clip(p_mkt[tr], 1e-4, 1 - 1e-4))
        lte_m, lte_k = lg(p_mod), lg(np.clip(pm, 1e-4, 1 - 1e-4))
        grid = np.linspace(0, 1, 101)
        losses = [log_loss(y[tr], 1 / (1 + np.exp(-(g * ltr_m + (1 - g) * ltr_k))))
                  for g in grid]
        lam = float(grid[int(np.argmin(losses))])
        cm, ck = lam, 1 - lam
        p_bl = np.clip(1 / (1 + np.exp(-(lam * lte_m + (1 - lam) * lte_k))),
                       1e-6, 1 - 1e-6)
        ll_bl = log_loss(yy, p_bl)
        db = (-np.log(np.clip(np.where(yy == 1, pm, 1 - pm), 1e-9, 1))
              - -np.log(np.clip(np.where(yy == 1, p_bl, 1 - p_bl), 1e-9, 1)))
        blo, bhi = bootstrap(db)
        print(f"\n  БЛЕНД   λ модели {cm:.2f} · рынка {ck:.2f}")
        print(f"          log-loss {ll_bl:.4f} против рынка {ll_mkt:.4f} · "
              f"{db.mean():+.4f} [95% {blo:+.4f}, {bhi:+.4f}]")
        print("          вывод:", "БЛЕНД БЬЁТ РЫНОК" if blo > 0 else
              ("рынок не улучшить" if bhi < 0 else "неотличимо от нуля"))
        out |= {"blend_w_model": float(cm), "blend_w_market": float(ck),
                "ll_blend": float(ll_bl), "blend_gap": float(db.mean()),
                "blend_ci": [float(blo), float(bhi)]}

    res = CACHE / "market_eval_results.jsonl"
    with res.open("a") as fh:
        fh.write(json.dumps(out) + "\n")

    # Keep the per-bout predictions so two feature sets can be compared to each
    # other by a PAIRED bootstrap on the same bouts, not by eyeballing two
    # log-losses that each carry their own ±0.03.
    preds = CACHE / "preds"
    preds.mkdir(exist_ok=True)
    np.savez(preds / f"{label}.npz", p=p_mod, y=yy, p_mkt=pm,
             key=j.loc[te, "index"].to_numpy(),
             # raw decimal prices, vig included — log-loss is scored on the
             # devigged probability, but a bet is settled at the real number
             ca=ca[te], cb=cb[te],
             p_blend=(p_bl if "--blend" in sys.argv else p_mod),
             p_corp=p_corp, y_corp=y_corp, key_corp=pidx)


if __name__ == "__main__":
    main()
