"""The bench: one data load, many variants, one paired verdict each.

market_eval.py is the scoreboard and it re-reads and re-joins everything on
every run, which is right for a headline and wrong for a search. This loads the
corpus, the feature matrix and the odds join ONCE and then trains as many
variants as asked, reporting each against the baseline by a PAIRED bootstrap on
the same bouts — the only comparison that can see 0.001.

Three instruments, all reported, because they answer different questions:
  corpus  89k post-cutoff bouts — the power to resolve a small feature gain
  prem    the 12.5k of them the market would have priced — the population
  quoted  the 3.3k actually priced — the only one that mentions the market

Every variant trains on exactly the same rows, with exactly the same split and
half-life as market_eval.py, so a difference here is the flag and nothing else.

  ./venv/bin/python scripts/lab.py --exp base,draws,levels --seeds 1
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(ROOT / "scripts" / "simulation" / "scripts"))
from src import features as F  # noqa: E402

import market_eval as ME  # noqa: E402

CACHE = ROOT / "imports" / "staging"
OUT = CACHE / "lab"
OUT.mkdir(exist_ok=True)


def ll(p, y):
    return -np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-9, 1))


def boot_ci(d, n=4000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    b = d[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ------------------------------------------------------------------ the mirror
def flip_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Every bout in the other corner order. winner_id and is_draw are facts
    about the men, not the corners, so they do not move."""
    out = df.copy()
    for x, y in F.PAIRED:
        if x in out.columns and y in out.columns:
            out[[x, y]] = out[[y, x]].values
    return out


def mirror_feats(tag: str, df: pd.DataFrame) -> pd.DataFrame:
    """The feature matrix as it would have been if every bout had been entered
    the other way round. Not derivable by negating columns — some of them are
    per-corner, some are ratios, and one wrong sign is a silent bug — so it is
    the same replay on the flipped frame, cached like the first one."""
    fc = CACHE / f"featsmir_{tag}_v{F.FEATS_VERSION}.parquet"
    if fc.exists():
        return pd.read_parquet(fc)
    t0 = time.time()
    print(f"  mirror replay of {len(df):,} bouts…", flush=True)
    fm = F.replay(flip_frame(df))
    fm.to_parquet(fc, index=False)
    print(f"  mirror replay done in {time.time() - t0:.0f}s", flush=True)
    return fm


# ------------------------------------------------------------------- the bench
class Bench:
    def __init__(self, tag: str = "card", price: str = "close") -> None:
        self.tag = tag
        t0 = time.time()
        self.df, self.feats = ME.build(tag)
        # float32 halves the matrix and costs LightGBM nothing — it bins every
        # column to 8 or 16 bits before the first split anyway. At 177 columns
        # the float64 copy plus its mirror was enough to put this machine into
        # swap, which showed up as a training run that never finished.
        self.feats = self.feats.astype("float32")
        self.y_all = F.label(self.df)
        j = ME.join_odds(self.df, verbose=False)
        same = ~j["swap"].to_numpy(bool)
        ca = np.where(same, j[f"{price}_a"], j[f"{price}_b"]).astype(float)
        cb = np.where(same, j[f"{price}_b"], j[f"{price}_a"]).astype(float)
        kk = np.isfinite(ca) & np.isfinite(cb) & (ca > 1) & (cb > 1)
        j, ca, cb = j[kk].reset_index(drop=True), ca[kk], cb[kk]
        self.jy = (j["winner_id"].astype(str) == j["a"].astype(str)).astype(int).values
        self.jidx = j["index"].values
        self.cutoff = pd.Series(j["dt"]).quantile(0.6)
        self.qtr = (j["dt"] <= self.cutoff).values
        self.qte = (j["dt"] > self.cutoff).values
        # the de-vig market_eval would have picked, fitted on the quoted train
        cands = {m: ME.devig(1 / ca, 1 / cb, m)
                 for m in ("proportional", "additive", "shin", "power")}
        slopes = {m: ME.devig_slope(v[self.qtr], self.jy[self.qtr])
                  for m, v in cands.items()}
        self.dv = min(slopes, key=lambda m: abs(slopes[m] - 1.0))
        self.p_mkt = cands[self.dv]
        # the raw decimal prices too: log-loss is scored on the devigged
        # probability, but a bet is settled at the number on the board
        self.ca, self.cb = ca, cb
        oa = np.where(same, j["open_a"], j["open_b"]).astype(float)
        ob = np.where(same, j["open_b"], j["open_a"]).astype(float)
        self.oa, self.ob = oa, ob

        pre = (self.df["dt"] <= self.cutoff).values
        nd = (~self.df["is_draw"]).values
        self.pre, self.nd = pre, nd
        self.big0 = np.where(nd & pre)[0]              # training rows, no draws
        self.draw_idx = np.where(self.df["is_draw"].values & pre)[0]
        self.post = np.where(nd & ~pre)[0]             # corpus holdout
        self.prem_all = ((np.nan_to_num(self.feats["sched_rounds"].to_numpy(), nan=0) >= 8)
                         & (np.minimum(self.feats["n_a"].to_numpy(),
                                       self.feats["n_b"].to_numpy()) >= 8))
        self.prem = self.prem_all[self.post]
        self.dt = self.df["dt"].to_numpy("datetime64[D]")
        # Choosing a variant on the same 89k bouts that then report it is a
        # selection tax paid in silence. Split the holdout in half by date: the
        # EARLY half decides, the LATE half reports, and nothing is ever chosen
        # on the late half. Both are large enough to resolve 0.002.
        pdt = self.df["dt"].to_numpy()[self.post]
        mid = np.quantile(pdt.astype("datetime64[D]").astype(np.int64), 0.5)
        self.sel = pdt.astype("datetime64[D]").astype(np.int64) <= mid
        self.conf = ~self.sel
        self._mir: pd.DataFrame | None = None
        print(f"bench ready in {time.time() - t0:.0f}s · cutoff {self.cutoff.date()} · "
              f"train {len(self.big0):,} (+{len(self.draw_idx):,} draws) · "
              f"corpus holdout {len(self.post):,} "
              f"(select {int(self.sel.sum()):,} / confirm {int(self.conf.sum()):,}) · "
              f"premium {int(self.prem.sum()):,} · "
              f"quoted test {int(self.qte.sum()):,} · de-vig {self.dv}", flush=True)

    @property
    def mir(self) -> pd.DataFrame:
        if self._mir is None:
            self._mir = mirror_feats(self.tag, self.df).astype("float32")
        return self._mir


# ------------------------------------------------------------------- one train
def fit(bench: Bench, cols: list[str], cutoff, *, seeds: int = 1,
        halflife: float = 6.0, draws: bool = False, mirror_train: bool = False,
        tta: bool = False, weight: str = "none", params_over: dict | None = None,
        finetune: int = 0, refit: bool = False, valfrac: float = 0.1,
        init_score: bool = False, seed0: int = 42):
    """Train on everything up to `cutoff` and hand back a predictor."""
    import lightgbm as lgb

    B = bench
    pre = B.dt <= np.datetime64(cutoff, "D")
    big = np.where(B.nd & pre)[0]
    y_big = B.y_all[big].astype(float)
    w_big = np.ones(len(big))
    if draws:
        d = np.where(B.df["is_draw"].to_numpy() & pre)[0]
        big = np.concatenate([big, d, d])
        y_big = np.concatenate([y_big, np.ones(len(d)), np.zeros(len(d))])
        w_big = np.concatenate([w_big, np.full(len(d), 0.5), np.full(len(d), 0.5)])
        order = np.argsort(B.dt[big], kind="stable")
        big, y_big, w_big = big[order], y_big[order], w_big[order]
    if weight == "quoted":
        w_big = w_big * np.where(B.prem_all[big], 3.0, 1.0)
    elif weight.startswith("top"):
        # Tonight's finding says the edge lives on twelve-rounders and title
        # fights. If those are the only bouts we would ever bet, accuracy on the
        # four-round undercard is not worth a single split. This is NOT the
        # premium reweighting that failed — that one was "sched >= 8 and both
        # men 8 bouts", which is most of the quoted set; this is the top of it.
        mult = float(weight[3:]) if len(weight) > 3 else 4.0
        sch = np.nan_to_num(B.feats["sched_rounds"].to_numpy()[big], nan=0)
        tit = np.nan_to_num(B.feats["title_lvl"].to_numpy()[big], nan=0)
        w_big = w_big * np.where((sch >= 12) | (tit >= 3), mult, 1.0)
    elif weight.startswith("comp"):
        # Nine tenths of the training corpus is a padded prospect against a
        # journeyman, where the answer is known before the bell and there is
        # nothing to learn. Weight a bout by how close the ratings said it was —
        # 4p(1−p) from the Elo-implied probability, which is 1 at a coin flip
        # and falls to 0 at a mismatch — with a floor so the easy bouts still
        # teach the model what a certainty looks like.
        #
        # This is NOT the quoted-population reweighting that failed. That one
        # asked "is this the kind of card the market prices"; this asks "was
        # this fight in doubt", which is the axis the model is actually weak on.
        floor = float(weight[4:]) if len(weight) > 4 else 0.25
        pe = 1.0 / (1.0 + 10 ** (-B.feats["d_elo"].to_numpy()[big] / 400.0))
        comp = 4.0 * pe * (1.0 - pe)
        w_big = w_big * (floor + (1.0 - floor) * np.nan_to_num(comp, nan=0.5))
    if halflife > 0:
        yrs = (np.datetime64(cutoff, "D") - B.dt[big]) / np.timedelta64(365, "D")
        w_big = w_big * 0.5 ** (np.clip(yrs, 0, None) / halflife)

    X = B.feats[cols]
    cut = int(len(big) * (1.0 - valfrac))
    itr, iva = big[:cut], big[cut:]
    Xtr, ytr, wtr = X.iloc[itr], y_big[:cut], w_big[:cut]
    Xva, yva, wva = X.iloc[iva], y_big[cut:], w_big[cut:]
    X_mir = B.mir[cols] if (mirror_train or tta) else None
    if mirror_train:
        # the same bouts entered the other way round: exact antisymmetry as
        # augmentation, and the mirror of a training bout stays on the training
        # side of the split so it can never be validated against its own twin
        M = X_mir
        Xtr = pd.concat([Xtr, M.iloc[itr]], ignore_index=True)
        ytr = np.concatenate([ytr, 1.0 - ytr])
        wtr = np.concatenate([wtr, wtr])
        Xva = pd.concat([Xva, M.iloc[iva]], ignore_index=True)
        yva = np.concatenate([yva, 1.0 - yva])
        wva = np.concatenate([wva, wva])

    params = {"objective": "binary", "metric": "binary_logloss",
              "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 100,
              "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
              "lambda_l2": 5.0, "verbosity": -1, "seed": seed0, "num_threads": 8}
    params.update(params_over or {})
    # Glickman's own expected score as the starting point, so boosting only has
    # to learn what the ratings do NOT explain. Different from handing it over
    # as a feature: as a feature the trees may ignore it, as an offset the
    # uncertainty deflation is in the answer whether they use it or not.
    def _z0(rows, mirror=False):
        # read off the full matrix, not the selected columns: the offset must
        # exist even when glicko_e itself is not in the feature set
        src = B.mir if mirror else B.feats
        g = np.clip(src["glicko_e"].to_numpy()[rows], 1e-4, 1 - 1e-4)
        return np.log(g / (1 - g)).astype(float)

    if init_score:
        z0tr, z0va = _z0(itr), _z0(iva)
        if mirror_train:
            z0tr = np.concatenate([z0tr, -z0tr])
            z0va = np.concatenate([z0va, -z0va])
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr,
                      init_score=z0tr if init_score else None)
    dva = lgb.Dataset(Xva, label=yva, weight=wva, reference=dtr,
                      init_score=z0va if init_score else None)
    # Bin now and drop our own copy. With --mirror the training frame is 612k
    # rows by 200 columns and it is held alongside the base matrix, the mirror,
    # and both of their column subsets — enough to put a 16GB machine into swap,
    # where a five-seed run stops looking slow and starts looking hung.
    dtr.construct(); dva.construct()
    if not finetune:                 # the fine-tune path still reads the frames
        del Xtr, Xva
    models = []
    for k in range(seeds):
        p = dict(params, seed=seed0 + k, bagging_seed=seed0 + k,
                 feature_fraction_seed=seed0 + k)
        models.append(lgb.train(p, dtr, num_boost_round=4000, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(150, verbose=False)]))
    if finetune:
        pm_tr, pm_va = B.prem_all[itr], B.prem_all[iva]
        if mirror_train:
            pm_tr = np.concatenate([pm_tr, pm_tr])
            pm_va = np.concatenate([pm_va, pm_va])
        tuned = []
        for k, m in enumerate(models):
            ptr = lgb.Dataset(Xtr[pm_tr], label=ytr[pm_tr], weight=wtr[pm_tr])
            pva = lgb.Dataset(Xva[pm_va], label=yva[pm_va], weight=wva[pm_va],
                              reference=ptr)
            fp = dict(params, learning_rate=0.01, seed=seed0 + k,
                      bagging_seed=seed0 + k, feature_fraction_seed=seed0 + k)
            tuned.append(lgb.train(fp, ptr, num_boost_round=finetune,
                                   valid_sets=[pva], init_model=m,
                                   callbacks=[lgb.early_stopping(100, verbose=False)]))
        models = tuned

    if refit:
        # Early stopping spends the most recent tenth of the corpus on choosing
        # a tree count and then never trains on it — and with a six-year
        # half-life that tenth is the most heavily weighted data we have. Take
        # the tree count it found, put the validation slice back, and refit on
        # everything. The count is scaled by 1/0.9 because there is now that
        # much more data for the same number of passes.
        n = max(int(round(np.mean([m.best_iteration for m in models])
                          / (1.0 - valfrac))), 50)
        Xa, ya, wa_ = X.iloc[big], y_big, w_big
        if mirror_train:
            Xa = pd.concat([Xa, X_mir.iloc[big]], ignore_index=True)
            ya = np.concatenate([ya, 1.0 - ya])
            wa_ = np.concatenate([wa_, wa_])
        z0a = None
        if init_score:
            z0a = _z0(big)
            if mirror_train:
                z0a = np.concatenate([z0a, -z0a])
        dall = lgb.Dataset(Xa, label=ya, weight=wa_, init_score=z0a)
        models = [lgb.train(dict(params, seed=seed0 + k, bagging_seed=seed0 + k,
                                 feature_fraction_seed=seed0 + k),
                            dall, num_boost_round=n) for k in range(seeds)]
        for m in models:
            m.best_iteration = n

    def predict(rows: np.ndarray) -> np.ndarray:
        def _p(Z):
            ps = [np.clip(m.predict(Z, num_iteration=m.best_iteration), 1e-6, 1 - 1e-6)
                  for m in models]
            return np.mean([np.log(p / (1 - p)) for p in ps], axis=0)
        lg = _p(X.iloc[rows])
        if init_score:
            lg = lg + _z0(rows)          # the offset is not in Booster.predict
        if tta:
            # the same bout asked the other way round; the model is not exactly
            # antisymmetric, and the half of the disagreement that is noise
            # cancels
            lgm = _p(X_mir.iloc[rows])
            if init_score:
                lgm = lgm + _z0(rows, mirror=True)
            lg = 0.5 * (lg - lgm)
        return np.clip(1 / (1 + np.exp(-lg)), 1e-6, 1 - 1e-6)

    predict.n_trees = float(np.mean([m.best_iteration for m in models]))
    return predict


def run(bench: Bench, cols: list[str], *, walk_months: int = 0,
        hl_mix: list | None = None, extra: dict | None = None, **kw) -> dict:
    """Score one configuration on all three instruments.

    walk_months > 0 retrains at that cadence through the holdout instead of
    predicting three years of it from one model fitted in 2023. It is not a
    trick: it is what deployment would do, and the difference is the price of
    staleness — measured rather than assumed.
    """
    B = bench
    if hl_mix:
        # A half-life is a claim about how fast a bout stops being evidence, and
        # six years is a compromise between two different claims: three years
        # tracks a fighter who is changing, twelve remembers a division. Rather
        # than pick, fit one model per half-life and average the logits — the
        # same trick as seed bagging, but the members disagree for a reason
        # instead of by accident, which is what makes an average worth more than
        # its best member.
        lgs, trees = [], []
        for h in hl_mix:
            pr = fit(B, cols, B.cutoff, halflife=h, **kw)
            lgs.append((np.log(pr(B.post) / (1 - pr(B.post))),
                        np.log(pr(B.jidx[B.qte]) / (1 - pr(B.jidx[B.qte])))))
            trees.append(pr.n_trees)
        p_corp = 1 / (1 + np.exp(-np.mean([a for a, _ in lgs], axis=0)))
        p_q = 1 / (1 + np.exp(-np.mean([b for _, b in lgs], axis=0)))
        n_trees = float(np.mean(trees))
    elif walk_months <= 0:
        pr = fit(B, cols, B.cutoff, **kw)
        p_corp, p_q, n_trees = pr(B.post), pr(B.jidx[B.qte]), pr.n_trees
    else:
        edges = [B.cutoff]
        while edges[-1] < pd.Timestamp(B.dt.max()):
            edges.append(edges[-1] + pd.DateOffset(months=walk_months))
        p_corp = np.zeros(len(B.post))
        qrows = B.jidx[B.qte]
        p_q = np.zeros(len(qrows))
        trees, nseg = [], 0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mc = (B.dt[B.post] > np.datetime64(lo, "D")) & (B.dt[B.post] <= np.datetime64(hi, "D"))
            mq = (B.dt[qrows] > np.datetime64(lo, "D")) & (B.dt[qrows] <= np.datetime64(hi, "D"))
            if not mc.any() and not mq.any():
                continue
            pr = fit(B, cols, lo, **kw)
            if mc.any():
                p_corp[mc] = pr(B.post[mc])
            if mq.any():
                p_q[mq] = pr(qrows[mq])
            trees.append(pr.n_trees); nseg += 1
        assert (p_corp > 0).all() and (p_q > 0).all(), "a holdout bout got no model"
        n_trees = float(np.mean(trees))
    y_corp = B.y_all[B.post]
    from sklearn.metrics import log_loss
    res = {"n_trees": int(n_trees),
           "ll_corpus": float(log_loss(y_corp, p_corp)),
           "ll_select": float(log_loss(y_corp[B.sel], p_corp[B.sel])),
           "ll_confirm": float(log_loss(y_corp[B.conf], p_corp[B.conf])),
           "ll_prem": float(log_loss(y_corp[B.prem], p_corp[B.prem])),
           "ll_quoted": float(log_loss(B.jy[B.qte], p_q)),
           "ll_market": float(log_loss(B.jy[B.qte],
                                       np.clip(B.p_mkt[B.qte], 1e-6, 1 - 1e-6))),
           "n_feats": len(cols)}
    res.update(extra or {})
    res["_p_corp"], res["_y_corp"] = p_corp, y_corp
    res["_p_q"], res["_y_q"] = p_q, B.jy[B.qte]
    return res


def report(name: str, r: dict, base: dict | None) -> None:
    line = (f"{name:22s} corpus {r['ll_corpus']:.4f}  sel {r['ll_select']:.4f}  "
            f"conf {r['ll_confirm']:.4f}  prem {r['ll_prem']:.4f}  "
            f"quoted {r['ll_quoted']:.4f} (mkt {r['ll_market']:.4f})  "
            f"{r['n_feats']}f {r['n_trees']}t")
    if base is not None and base is not r:
        masks = [("select", BENCH.sel), ("confirm", BENCH.conf), ("prem", BENCH.prem)]
        for tag, m in masks:
            d = (ll(base["_p_corp"][m], base["_y_corp"][m])
                 - ll(r["_p_corp"][m], r["_y_corp"][m]))
            lo, hi = boot_ci(d)
            mark = "+" if lo > 0 else ("-" if hi < 0 else " ")
            line += f"\n{'':22s}  Δ{tag:8s} {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {mark}"
        d = ll(base["_p_q"], base["_y_q"]) - ll(r["_p_q"], r["_y_q"])
        lo, hi = boot_ci(d)
        line += f"\n{'':22s}  Δquoted   {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]"
    print(line, flush=True)


BENCH: Bench | None = None


def table() -> None:
    """Every variant ever run on this bench, ranked by the confirmation half —
    the one no variant was ever chosen on."""
    s = json.loads((OUT / "summary.json").read_text())
    base = s.get("base", {})
    rows = sorted(s.items(), key=lambda kv: kv[1]["ll_confirm"])
    print(f"{'variant':22s} {'corpus':>8s} {'select':>8s} {'confirm':>8s} "
          f"{'prem':>8s} {'quoted':>8s} {'vs base (conf)':>15s}")
    for k, v in rows:
        d = base.get("ll_confirm", v["ll_confirm"]) - v["ll_confirm"]
        print(f"{k:22s} {v['ll_corpus']:8.4f} {v['ll_select']:8.4f} "
              f"{v['ll_confirm']:8.4f} {v['ll_prem']:8.4f} {v['ll_quoted']:8.4f} "
              f"{d:+15.4f}")


def main() -> None:
    global BENCH
    if "--table" in sys.argv:
        return table()
    seeds = int(ME.arg("--seeds", "1"))
    fset = ME.arg("--feats", "everyc")
    only = ME.arg("--exp", "")
    BENCH = Bench(ME.arg("--tag", "card"))
    cols = ME.resolve(fset)

    # name -> kwargs for run(); "base" must stay first and unchanged.
    # a "feats" key names a different feature set for that variant only.
    EXPS: dict[str, dict] = {
        "base": {},
        # the same configuration on a different seed: whatever it differs from
        # base by is the resolution of a one-seed screen, and no variant closer
        # than that has been measured at all
        "base-seedB": {"seed0": 1042},
        "base-seedC": {"seed0": 2042},
        "walk12": {"walk_months": 12},
        "walk6": {"walk_months": 6},
        "walk24": {"walk_months": 24},
        "refit": {"refit": True},
        "refit+walk12": {"refit": True, "walk_months": 12},
        "val03": {"valfrac": 0.03},
        "initscore": {"init_score": True},
        "initscore+tta": {"init_score": True, "tta": True},
        "refit-val03": {"refit": True, "valfrac": 0.03},
        "seeds10": {"seeds": 10},
        "seeds3": {"seeds": 3},
        "x-all": {"feats": "everyx"},
        "x-lvlr": {"feats": "everyc+lvlr"},
        "x-lvlq": {"feats": "everyc+lvlq"},
        "x-unc": {"feats": "everyc+unc"},
        "x-res": {"feats": "everyc+res"},
        "x-ctx": {"feats": "everyc+ctx"},
        "x-lvl": {"feats": "everyc+lvlr+lvlq"},
        "x-no-lvlr": {"feats": "everyx", "drop": "lvlr"},
        "x-no-lvlq": {"feats": "everyx", "drop": "lvlq"},
        "x-no-unc": {"feats": "everyx", "drop": "unc"},
        "x-no-res": {"feats": "everyx", "drop": "res"},
        "x-no-ctx": {"feats": "everyx", "drop": "ctx"},
        "x-no-cmp": {"feats": "everyx", "drop": "cmp"},
        "x-no-thin": {"feats": "everyx", "drop": "thin"},
        "x-no-amat": {"feats": "everyx", "drop": "amat"},
        # everyx minus the amateur group, plus its single strongest column.
        # Spelled out rather than expressed as a drop: --drop runs AFTER
        # resolve, so "everyx+amat1 drop amat" removes d_am again and
        # silently reproduces x-no-amat.
        "x-amat1": {"feats": "everyc+lvlr+lvlq+unc+res+ctx+cmp+thin+amat1"},
        "x-amat": {"feats": "everyc+amat"},
        "amat-tta": {"feats": "everyx", "tta": True},
        "noamat-tta": {"feats": "everyx", "drop": "amat", "tta": True},
        "x-thin": {"feats": "everyc+thin"},
        "x-trim": {"feats": "everyx", "drop": "lvlr+lvlq"},
        "x-cmp": {"feats": "everyc+cmp"},
        # the ladder: each rung adds one thing to the rung below it, so the
        # paired bootstrap between neighbours is what that one thing is worth
        "L1-feats": {"feats": "everyx"},
        "L2-tta": {"feats": "everyx", "tta": True},
        "L3-mirror": {"feats": "everyx", "mirror_train": True, "tta": True},
        "L4-walk": {"feats": "everyx", "mirror_train": True, "tta": True,
                    "walk_months": 12},
        "draws": {"draws": True},
        "tta": {"tta": True},
        "mirror": {"mirror_train": True},
        "mirror+tta": {"mirror_train": True, "tta": True},
        "draws+tta": {"draws": True, "tta": True},
        "draws+mirror+tta": {"draws": True, "mirror_train": True, "tta": True},
        "finetune": {"finetune": 400},
        "wquoted": {"weight": "quoted"},
        "wcomp25": {"weight": "comp0.25", "tta": True},
        "wcomp50": {"weight": "comp0.50", "tta": True},
        "wcomp10": {"weight": "comp0.10", "tta": True},
        "tta-ref": {"tta": True},
        "wtop4": {"weight": "top4", "tta": True},
        "wtop10": {"weight": "top10", "tta": True},
        # five seeds were chosen when each was worth ~+0.0013 going 1->5.
        # Nobody has asked what fifteen buys; it is pure compute.
        "tta-s15": {"tta": True, "seeds": 15},
        "walk-tta": {"tta": True, "walk_months": 12},
        "mirror-tta": {"mirror_train": True, "tta": True},
        "mirror-walk": {"mirror_train": True, "tta": True,
                        "walk_months": 12},
        "hlmix": {"tta": True, "hl_mix": [3.0, 6.0, 12.0]},
        "hlmix-mirror": {"mirror_train": True, "tta": True,
                         "hl_mix": [3.0, 6.0, 12.0]},
        "leaves127": {"params_over": {"num_leaves": 127}},
        "leaves31": {"params_over": {"num_leaves": 31}},
        "leaves255-lr02": {"params_over": {"num_leaves": 255, "learning_rate": 0.02}},
        "lr015": {"params_over": {"learning_rate": 0.015}},
        "minleaf300": {"params_over": {"min_data_in_leaf": 300}},
        "minleaf30": {"params_over": {"min_data_in_leaf": 30}},
        "l2-30": {"params_over": {"lambda_l2": 30.0}},
        "ff06": {"params_over": {"feature_fraction": 0.6}},
        "lineartree": {"params_over": {"linear_tree": True, "lambda_l2": 20.0}},
        "extratrees": {"params_over": {"extra_trees": True}},
        "hl4": {"halflife": 4.0},
        "hl9": {"halflife": 9.0},
        "hl0": {"halflife": 0.0},
    }
    names = [n for n in (only.split(",") if only else EXPS) if n in EXPS]
    if "base" not in names:
        names = ["base"] + names
    out = {}
    base = None
    for n in names:
        t0 = time.time()
        kw = dict(EXPS[n])
        c = cols
        if "feats" in kw or "drop" in kw:
            c = ME.resolve(kw.pop("feats", fset))
            if "drop" in kw:
                gone = {x for g in kw.pop("drop").split("+") for x in ME.GROUPS[g]}
                c = [x for x in c if x not in gone]
        r = run(BENCH, c, seeds=kw.pop("seeds", seeds), **kw)
        if base is None:
            base = r
        report(f"{n} ({time.time() - t0:.0f}s)", r, base)
        out[n] = {k: v for k, v in r.items() if not k.startswith("_")}
        np.savez(OUT / f"{n}.npz", p_corp=r["_p_corp"], y_corp=r["_y_corp"],
                 p_q=r["_p_q"], y_q=r["_y_q"], key_corp=BENCH.post,
                 prem=BENCH.prem)
    old = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    (OUT / "summary.json").write_text(json.dumps({**old, **out}, indent=1))


if __name__ == "__main__":
    main()
