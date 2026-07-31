"""Hyper-parameter search that cannot flatter itself.

The obvious way to do this is to try a thousand configurations and keep the one
with the lowest log-loss on the holdout. That number is then wrong, and wrong in
a direction we can compute: seed noise alone moves the premium holdout by about
0.0005, so the best of a thousand draws sits ~3.2 sigma below the mean, and
0.0016 of the "gain" is the search finding noise. Every real feature group
measured today was worth between 0.0014 and 0.0042. A search selected on the
reporting set would therefore be indistinguishable from the thing it is
supposed to measure.

So the search never sees the reporting set. Three windows, in time order:

    train        ... 2019-11   the tuning fit
    select   2019-11 .. 2022-11   what the search optimises
    report   2022-11 ..          untouched until one configuration is chosen

The real cutoff is the 60th percentile of the quoted bouts, exactly as in
market_eval; the tuning cutoff is three years earlier, so the whole search
happens inside data that the final evaluation does not contain. The search runs
with one seed for speed, the top few are re-ranked with five seeds on the SAME
selection window (that is where most of the noise is), and only the single
winner is ever measured on the report window.

  ./venv/bin/python scripts/tune.py --trials 400
  ./venv/bin/python scripts/tune.py --trials 0 --report   # re-report, no search
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(ROOT / "scripts" / "simulation" / "scripts"))
from src import features as F  # noqa: E402

CACHE = ROOT / "imports" / "staging"
TRIALS = CACHE / "tune_trials.jsonl"
STUDY = CACHE / "tune_study.db"


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(np.where(y == 1, np.log(p), np.log1p(-p))))


class Fold:
    """Everything a fit needs, built once. A trial is then a call to lgb."""

    def __init__(self, X, dt, y, prem, train_end, sel_end):
        pre = dt <= train_end
        self.big = np.where(pre)[0]
        self.y_tr = y[self.big]
        sel = (dt > train_end) & (dt <= sel_end) if sel_end is not None else (dt > train_end)
        self.sel = np.where(sel)[0]
        self.sel_prem = np.where(sel & prem)[0]
        self.X, self.dt, self.y = X, dt, y
        self.train_end = train_end
        cut = int(len(self.big) * 0.9)
        self.fit_idx, self.val_idx = self.big[:cut], self.big[cut:]
        self.prem = prem

    def weights(self, halflife: float, prem_w: float) -> np.ndarray:
        w = np.ones(len(self.big))
        if halflife > 0:
            yrs = ((np.datetime64(self.train_end) - self.dt[self.big])
                   / np.timedelta64(365, "D")).astype(float)
            w *= 0.5 ** (np.clip(yrs, 0, None) / halflife)
        if prem_w != 1.0:
            w *= np.where(self.prem[self.big], prem_w, 1.0)
        return w

    def fit(self, params, halflife, prem_w, seeds=1, rounds=None, trial=None):
        """One fit per seed. With a trial attached, the selection log-loss is
        reported every 100 rounds so a hopeless configuration is killed at 300
        instead of at 3,000 — half the search budget went to trials that were
        never going to win."""
        w = self.weights(halflife, prem_w)
        c = len(self.fit_idx)
        rounds = rounds or min(3000, int(50 / params["learning_rate"]))
        dtr = lgb.Dataset(self.X[self.fit_idx], label=self.y_tr[:c], weight=w[:c],
                          params=params, free_raw_data=False)
        dva = lgb.Dataset(self.X[self.val_idx], label=self.y_tr[c:], weight=w[c:],
                          reference=dtr, free_raw_data=False)
        cbs = [lgb.early_stopping(100, verbose=False)]
        if trial is not None:
            import optuna
            Xs, ys = self.X[self.sel_prem], self.y[self.sel_prem]

            def watch(env):
                if env.iteration % 100 or env.iteration < 200:
                    return
                p = np.clip(env.model.predict(Xs), 1e-6, 1 - 1e-6)
                trial.report(logloss(p, ys), env.iteration)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            cbs.append(watch)
        lg = []
        for k in range(seeds):
            p = dict(params, seed=42 + k, bagging_seed=42 + k,
                     feature_fraction_seed=42 + k)
            lg.append(lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                                callbacks=cbs))
        return lg

    def predict(self, models, idx):
        ps = [np.clip(m.predict(self.X[idx], num_iteration=m.best_iteration),
                      1e-6, 1 - 1e-6) for m in models]
        z = np.mean([np.log(p / (1 - p)) for p in ps], axis=0)
        return 1.0 / (1.0 + np.exp(-z))


SPACE = {
    # the learner
    # below 0.012 a trial spends three minutes to arrive somewhere worse — the
    # four-trial timing probe cost 184s at lr=0.0063 and scored 0.3331
    "learning_rate": ("logf", 0.012, 0.10),
    "num_leaves": ("logi", 15, 511),
    "min_data_in_leaf": ("logi", 20, 1000),
    "feature_fraction": ("f", 0.30, 1.0),
    "bagging_fraction": ("f", 0.50, 1.0),
    "bagging_freq": ("i", 0, 10),
    "lambda_l1": ("logf", 1e-3, 20.0),
    "lambda_l2": ("logf", 1e-3, 100.0),
    "min_gain_to_split": ("f", 0.0, 1.0),
    "min_sum_hessian_in_leaf": ("logf", 1e-3, 20.0),
    "path_smooth": ("logf", 1e-3, 200.0),
    "max_depth": ("c", [-1, 5, 7, 9, 12]),
    "max_bin": ("c", [63, 127, 255, 511]),
    "extra_trees": ("c", [False, True]),
    # the sample, which is not a LightGBM parameter and mattered more than most
    "halflife": ("c", [0.0, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0, 20.0, 40.0]),
    "prem_w": ("f", 1.0, 6.0),
}


def suggest(trial):
    out = {}
    for k, spec in SPACE.items():
        kind = spec[0]
        if kind == "f":
            out[k] = trial.suggest_float(k, spec[1], spec[2])
        elif kind == "logf":
            out[k] = trial.suggest_float(k, spec[1], spec[2], log=True)
        elif kind == "i":
            out[k] = trial.suggest_int(k, spec[1], spec[2])
        elif kind == "logi":
            out[k] = trial.suggest_int(k, spec[1], spec[2], log=True)
        else:
            out[k] = trial.suggest_categorical(k, spec[1])
    return out


def to_lgb(cfg):
    p = {k: v for k, v in cfg.items() if k not in ("halflife", "prem_w")}
    p |= {"objective": "binary", "metric": "binary_logloss", "verbosity": -1,
          "num_threads": 0, "force_row_wise": True}
    if p["bagging_freq"] == 0:
        p["bagging_fraction"] = 1.0
    return p


def main() -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import market_eval as ME

    tag = arg("--tag", "weigh")
    n_trials = int(arg("--trials", "400"))
    df = pd.read_parquet(CACHE / f"sym_{tag}.parquet")
    feats = pd.read_parquet(CACHE / f"feats_{tag}_v{F.FEATS_VERSION}.parquet")
    j = ME.join_odds(df, verbose=False)
    real_cut = pd.Series(j["dt"]).quantile(0.6)
    tune_cut = real_cut - pd.DateOffset(years=int(arg("--back", "3")))

    keep = (~df["is_draw"]).to_numpy()
    X = feats[F.EVERY_W].to_numpy(np.float32)[keep]
    dt = df["dt"].to_numpy("datetime64[D]")[keep]
    y = F.label(df)[keep].astype(float)
    prem = ((np.nan_to_num(feats["sched_rounds"].to_numpy(), nan=0) >= 8)
            & (np.minimum(feats["n_a"].to_numpy(), feats["n_b"].to_numpy()) >= 8))[keep]

    tune = Fold(X, dt, y, prem, np.datetime64(tune_cut), np.datetime64(real_cut))
    print(f"подбор: обучение на {len(tune.big):,} боях до {tune_cut.date()} · "
          f"отбор на {len(tune.sel_prem):,} премиальных из "
          f"{len(tune.sel):,} за {tune_cut.date()}..{real_cut.date()}")
    print(f"отчёт:  холдаут после {real_cut.date()} НЕ ТРОГАЕТСЯ до конца поиска\n")

    if n_trials:
        fh = TRIALS.open("a")
        t_start = time.time()

        def objective(trial):
            cfg = suggest(trial)
            t0 = time.time()
            models = tune.fit(to_lgb(cfg), cfg["halflife"], cfg["prem_w"],
                              seeds=1, trial=trial)
            p = tune.predict(models, tune.sel_prem)
            ll = logloss(p, y[tune.sel_prem])
            llc = logloss(tune.predict(models, tune.sel), y[tune.sel])
            rec = {"n": trial.number, "ll_prem": ll, "ll_corp": llc,
                   "trees": models[0].best_iteration, "secs": round(time.time() - t0, 1),
                   **{k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                          else v) for k, v in cfg.items()}}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if trial.number % 10 == 0:
                done = [t for t in trial.study.trials if t.value is not None]
                best = min([t.value for t in done], default=ll)
                print(f"  {trial.number:4d} · {ll:.4f} лучший {min(best, ll):.4f}"
                      f" · {(time.time() - t_start) / 60:.0f} мин", flush=True)
            return ll

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=7),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=15,
                                               n_warmup_steps=300, interval_steps=100))
        # seed the search with what the one-knob-at-a-time sweep already found
        study.enqueue_trial({"learning_rate": 0.03, "num_leaves": 63,
                             "min_data_in_leaf": 100, "feature_fraction": 0.9,
                             "bagging_fraction": 0.9, "bagging_freq": 5,
                             "lambda_l1": 1e-3, "lambda_l2": 5.0,
                             "min_gain_to_split": 0.0, "min_sum_hessian_in_leaf": 1e-3,
                             "path_smooth": 1e-3, "max_depth": -1, "max_bin": 255,
                             "extra_trees": False, "halflife": 6.0, "prem_w": 1.0})
        study.optimize(objective, n_trials=n_trials)
        fh.close()

    # ---- re-rank the shortlist with five seeds, still on the selection window
    rows = [json.loads(l) for l in TRIALS.open()] if TRIALS.exists() else []
    rows.sort(key=lambda r: r["ll_prem"])
    short = rows[:int(arg("--short", "8"))]
    print(f"\nпереоценка {len(short)} лучших пятью сидами на ТОМ ЖЕ окне отбора:")
    for r in short:
        cfg = {k: r[k] for k in SPACE}
        cfg["num_leaves"] = int(cfg["num_leaves"]); cfg["max_bin"] = int(cfg["max_bin"])
        cfg["min_data_in_leaf"] = int(cfg["min_data_in_leaf"])
        cfg["bagging_freq"] = int(cfg["bagging_freq"]); cfg["max_depth"] = int(cfg["max_depth"])
        m = tune.fit(to_lgb(cfg), cfg["halflife"], cfg["prem_w"], seeds=5)
        r["ll5"] = logloss(tune.predict(m, tune.sel_prem), y[tune.sel_prem])
        print(f"  проба {r['n']:4d}: 1 сид {r['ll_prem']:.4f} → 5 сидов {r['ll5']:.4f}"
              f" · lr {r['learning_rate']:.4f} leaves {int(r['num_leaves'])} "
              f"minleaf {int(r['min_data_in_leaf'])} hl {r['halflife']}")
    short.sort(key=lambda r: r["ll5"])
    win = short[0]
    print(f"\nпобедитель: проба {win['n']}")
    print(json.dumps({k: win[k] for k in SPACE}, indent=2))

    # ---- one measurement on the window nothing has touched
    rep = Fold(X, dt, y, prem, np.datetime64(real_cut), None)
    cfg = {k: win[k] for k in SPACE}
    for k in ("num_leaves", "min_data_in_leaf", "bagging_freq", "max_bin", "max_depth"):
        cfg[k] = int(cfg[k])
    m = rep.fit(to_lgb(cfg), cfg["halflife"], cfg["prem_w"], seeds=5)
    ll_p = logloss(rep.predict(m, rep.sel_prem), y[rep.sel_prem])
    ll_c = logloss(rep.predict(m, rep.sel), y[rep.sel])
    print(f"\nОТЧЁТ, один раз, на нетронутом холдауте после {real_cut.date()}:")
    print(f"  корпус  n={len(rep.sel):,}: {ll_c:.4f}   (текущая модель 0.3548)")
    print(f"  премиум n={len(rep.sel_prem):,}: {ll_p:.4f}   (текущая модель 0.3025)")
    print(f"  налог за отбор: подбор обещал {win['ll5']:.4f} на своём окне")
    np.savez(CACHE / "preds" / "tuned.npz",
             p_corp=rep.predict(m, rep.sel), y_corp=y[rep.sel], key_corp=rep.sel,
             prem_corp=prem[rep.sel])
    (CACHE / "tune_winner.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
