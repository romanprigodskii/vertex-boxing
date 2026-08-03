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
    fc = CACHE / F.cache_name("featsmir", tag)
    if fc.exists():
        return pd.read_parquet(fc)
    t0 = time.time()
    print(f"  mirror replay of {len(df):,} bouts…", flush=True)
    fm = F.replay(flip_frame(df))
    fm.to_parquet(fc, index=False)
    print(f"  mirror replay done in {time.time() - t0:.0f}s", flush=True)
    return fm


# ------------------------------------------------- the axes a tree cannot build
# Twenty of the 200 columns reach the model as a PAIR — one number per corner —
# rather than as a difference. Verified against the mirror matrix on all 413,279
# rows: 82 columns negate under a corner swap, 96 are invariant, two go to 1-x,
# and exactly these ten pairs swap with each other.
#
# A boosting tree splits on one column at a time, so it can never form x_a - x_b
# on its own; it approximates the diagonal with a staircase and spends depth
# doing it. Four of the ten already have an exact difference column elsewhere in
# the matrix (d_bouts, d_bouts_true, d_age, d_over) — the other six do not, and
# for those the model has no antisymmetric axis at all.
#
# The sum is added too: it is the corner-invariant half of the same rotation and
# says "how much of this quantity is in the ring", which min/max covers for two
# of the pairs and for none of the rest. Both are exactly mirror-consistent by
# construction — computed with the same expression on the mirrored frame, the
# sum is unchanged and the difference negates.
PAIRS = [("n_a", "n_b", "nb", True), ("n_true_a", "n_true_b", "ntrue", True),
         ("age_a", "age_b", "age", True), ("over_a", "over_b", "over", True),
         ("lay_a", "lay_b", "lay", False),
         ("a_unbeaten", "b_unbeaten", "unb", False),
         ("rd_a", "rd_b", "rd", False), ("stepup_a", "stepup_b", "step", False),
         ("btn_a", "btn_b", "btn", False), ("dom_n_a", "dom_n_b", "domn", False)]


def ensure_rot(B) -> list[str]:
    """Add (x_a - x_b, x_a + x_b) for every per-corner pair. Returns the names."""
    names: list[str] = []
    for xa, xb, stem, has_diff in PAIRS:
        if xa not in B.feats.columns or xb not in B.feats.columns:
            continue
        cs, cd = f"{stem}_sum", f"{stem}_dif"
        for frame in (B.feats, B.mir):
            a = frame[xa].to_numpy("float32")
            b = frame[xb].to_numpy("float32")
            if cs not in frame.columns:
                frame[cs] = a + b
            if not has_diff and cd not in frame.columns:
                frame[cd] = a - b
        names.append(cs)
        if not has_diff:
            names.append(cd)
    return names


STACK = ["stack_pstop", "stack_dom"]


def ensure_stack(B) -> list[str]:
    """The two auxiliary predictions from stack.py, added to both matrices.

    They are model outputs, not replay state, so they cannot live in features.py
    — but they must still mirror exactly, and here that is arithmetic rather
    than a second prediction: a stoppage is a fact about the fight and does not
    move when the corners are exchanged, and the dominance estimate is signed
    for corner A and negates. stack.py already symmetrised both.
    """
    p = CACHE / f"stack_{B.tag}{F.tune_tag()}_v{F.FEATS_VERSION}.parquet"
    if not p.exists():
        raise SystemExit(f"{p.name} is missing — run "
                         f"scripts/stack.py --tag {B.tag} first")
    s = pd.read_parquet(p)
    assert len(s) == len(B.feats), "the stack file is a different population"
    if "stack_pstop" not in B.feats.columns:
        for c in STACK:
            B.feats[c] = s[c].to_numpy("float32")
        B.mir["stack_pstop"] = s["stack_pstop"].to_numpy("float32")
        B.mir["stack_dom"] = (-s["stack_dom"]).to_numpy("float32")
    return list(STACK)


# ------------------------------------------------------------- a graded target
def soft_label(B, rows: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """The win/loss bit, graded by how decisive the win was.

    Every rating in src/features.py reads one bit per bout — Elo, Glicko and
    Bradley-Terry all see a win as a win — and features.py says so itself, which
    is why d_melo exists. The BOOSTING TARGET still reads one bit. The judges
    hand out a graded number on 144,337 bouts and the referee's stopping round
    grades 267,000 more, and none of it reaches the objective. A split decision
    over ten rounds is weaker evidence that A is the better fighter than a
    first-round knockout, and a 0/1 label says the two are identical evidence.

    tau in [0,1] is decisiveness; the label is (1-alpha)*y + alpha*(0.5 +
    0.5*tau signed by the winner), so alpha=0 is exactly the current target and
    the rung above it is a strict generalisation.

    The result is NOT a probability, and a model fitted to it is not calibrated.
    It is a monotone transform of one, which a two-parameter logistic fit on the
    validation slice undoes. That is not the calibration that status.md killed —
    that one re-calibrated an already-calibrated model and had nothing to do;
    this one is required by construction.
    """
    d = B.df.iloc[rows]
    meth = d["method"].astype(str).str.lower().to_numpy()
    rnd = pd.to_numeric(d["round_finished"], errors="coerce").to_numpy(float)
    sch = pd.to_numeric(d["sched"], errors="coerce").to_numpy(float)
    sa = pd.to_numeric(d["a_score"], errors="coerce").to_numpy(float)
    sb = pd.to_numeric(d["b_score"], errors="coerce").to_numpy(float)
    sch = np.where(np.isfinite(sch) & (sch > 0), sch, 10.0)

    # unknown method: the corpus mean decisiveness, so it neither sharpens nor
    # flattens a bout we know nothing about
    tau = np.full(len(rows), 0.55)
    stop = np.isin(meth, ["ko", "tko", "rtd"])
    # a stoppage is decisive, and the earlier it came the more decisive it was
    tau = np.where(stop, 0.90 - 0.30 * np.clip(rnd / sch, 0, 1), tau)
    tau = np.where(stop & ~np.isfinite(rnd), 0.75, tau)
    # a decision with the cards on it grades itself: points per scheduled round,
    # where a 10-9 sweep is 1.0 and anything past 1.5 is a shutout
    tot = sa + sb
    marg = np.where(np.isfinite(tot) & (tot > 0), 19.0 * np.abs(sa - sb) / tot, np.nan)
    has = np.isfinite(marg)
    dec = np.isin(meth, ["ud", "sd", "md", "pts", "technical_decision"])
    tau = np.where(dec & has, np.clip(marg / 1.5, 0.05, 0.90), tau)
    # a decision with no cards saved: the verdict itself is the only grade
    tau = np.where(dec & ~has & (meth == "ud"), 0.55, tau)
    tau = np.where(dec & ~has & np.isin(meth, ["sd", "md"]), 0.15, tau)
    tau = np.where(dec & ~has & (meth == "pts"), 0.45, tau)
    tau = np.where(dec & ~has & (meth == "technical_decision"), 0.30, tau)
    # a disqualification says nothing about who was better
    tau = np.where(np.isin(meth, ["dq", "nc"]), 0.0, tau)

    graded = np.where(y > 0.5, 0.5 + 0.5 * tau, 0.5 - 0.5 * tau)
    return np.clip((1.0 - alpha) * y + alpha * graded, 1e-4, 1 - 1e-4)


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
        init_score: bool = False, seed0: int = 42, blend: str = "",
        avg: str = "logit", soft: float = 0.0, shrink: bool = False,
        stop_on: str = ""):
    """Train on everything up to `cutoff` and hand back a predictor.

    `blend` adds a second model CLASS to the average rather than a second seed.
    Seed bagging averages members that differ by accident; two learners with
    different inductive biases disagree for a reason, and on a surface this
    smooth — a rating gap deflated by how much the ratings are trusted — an
    oblivious-tree learner and a leaf-wise one make different mistakes.
      cat      LightGBM and CatBoost, equal weight per member
      catonly  CatBoost alone, to see which half carries it

    `avg` is how members are combined. Logit averaging is the geometric mean of
    the odds and is what this bench has always done; it is sharper than the
    members and can be over-confident. Probability averaging is the mixture,
    and by Jensen its log-loss is at most the mean of the members'. Which one
    wins is an empirical question nobody here has asked.
    """
    import lightgbm as lgb

    assert not (refit and blend), \
        "refit rebuilds only the LightGBM members; the CatBoost half would be stale"
    assert not (refit and soft > 0), \
        "refit trains on iva, which the graded-target calibrator is fitted on"
    B = bench
    pre = B.dt <= np.datetime64(cutoff, "D")
    big = np.where(B.nd & pre)[0]
    y_big = B.y_all[big].astype(float)
    if soft > 0:
        assert not draws, "the graded target already says a draw is 0.5"
        y_big = soft_label(B, big, y_big, soft)
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
    # WHERE the tree count is chosen. Early stopping reads the last tenth of the
    # training window, and four fifths of that is club boxing the market never
    # prices — so the one number the fit picks for itself is picked on the
    # population we do not care about. This is NOT the training reweighting that
    # failed six times: the model still learns from every bout with the same
    # weight, and all that moves is the stopping criterion.
    if stop_on:
        pmv = B.prem_all[iva]
        mult = float(stop_on[4:]) if stop_on.startswith("prem") and len(stop_on) > 4 else 0.0
        wva = wva * np.where(pmv, 1.0, mult)
        if wva.sum() <= 0:
            raise SystemExit("stop_on left the validation slice with no weight")
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
              "lambda_l2": 5.0, "verbosity": -1, "seed": seed0, "num_threads": 8,
              # Without these two the same run is not the same run. LightGBM
              # picks row-wise or column-wise histogram building by TIMING a
              # few iterations, so a busier machine takes the other path and
              # sums the same floats in a different order; the trees then
              # diverge and the holdout moves by more than a feature group is
              # worth. Measured before they were set: two identical runs of
              # `base` scored 0.3406 and 0.3408 on the corpus. That is not seed
              # noise — the seed was the same — and it was silently underneath
              # every screen this bench has ever printed.
              "force_row_wise": True, "deterministic": True}
    if soft > 0:
        # "binary" wants a 0/1 label; cross_entropy is the same loss written for
        # a label anywhere in [0,1], which is what a graded outcome is
        params["objective"] = "cross_entropy"
        params["metric"] = "cross_entropy"
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
    # params must reach the Dataset, not only lgb.train: linear_tree, max_bin
    # and min_data_in_bin are decided when the matrix is binned, and passing
    # them later raises "Cannot change linear_tree after constructed Dataset
    # handle". Without this the `lineartree` variant could never run at all —
    # it is in the registry and has never once been measured.
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, params=params,
                      init_score=z0tr if init_score else None)
    dva = lgb.Dataset(Xva, label=yva, weight=wva, reference=dtr, params=params,
                      init_score=z0va if init_score else None)
    # Bin now and drop our own copy. With --mirror the training frame is 612k
    # rows by 200 columns and it is held alongside the base matrix, the mirror,
    # and both of their column subsets — enough to put a 16GB machine into swap,
    # where a five-seed run stops looking slow and starts looking hung.
    dtr.construct(); dva.construct()
    if not finetune and not blend:   # these paths still read the frames
        del Xtr, Xva
    models = []
    if blend != "catonly":
        for k in range(seeds):
            p = dict(params, seed=seed0 + k, bagging_seed=seed0 + k,
                     feature_fraction_seed=seed0 + k)
            models.append(lgb.train(p, dtr, num_boost_round=4000, valid_sets=[dva],
                                    callbacks=[lgb.early_stopping(150, verbose=False)]))
    cats = []
    if blend:
        # Symmetric oblivious trees: every node at a depth splits on the same
        # feature, which is a much stronger regulariser than LightGBM's
        # leaf-wise growth and fails differently. Depth 6 is CatBoost's own
        # default and is not tuned here — tuning it would be a search, and the
        # search on this problem has already been run and found nothing.
        from catboost import CatBoostClassifier, Pool
        # CatBoost's Logloss demands exactly two distinct label values and
        # raises on a graded target; CrossEntropy is its documented objective
        # for a label anywhere in [0,1], the exact counterpart of LightGBM's.
        closs = "CrossEntropy" if soft > 0 else "Logloss"
        ptr = Pool(Xtr, label=ytr, weight=wtr)
        pva = Pool(Xva, label=yva, weight=wva)
        for k in range(seeds):
            c = CatBoostClassifier(iterations=4000, learning_rate=0.03, depth=6,
                                   l2_leaf_reg=5.0, loss_function=closs,
                                   random_seed=seed0 + k, thread_count=8,
                                   od_type="Iter", od_wait=150, verbose=False,
                                   allow_writing_files=False)
            c.fit(ptr, eval_set=pva, use_best_model=True)
            cats.append(c)
        del ptr, pva
        if not finetune:
            del Xtr, Xva
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
        dall = lgb.Dataset(Xa, label=ya, weight=wa_, params=params, init_score=z0a)
        models = [lgb.train(dict(params, seed=seed0 + k, bagging_seed=seed0 + k,
                                 feature_fraction_seed=seed0 + k),
                            dall, num_boost_round=n) for k in range(seeds)]
        for m in models:
            m.best_iteration = n

    def _p(Z):
        ps = [np.clip(m.predict(Z, num_iteration=m.best_iteration), 1e-6, 1 - 1e-6)
              for m in models]
        ps += [np.clip(c.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6) for c in cats]
        if avg == "prob":
            q = np.clip(np.mean(ps, axis=0), 1e-6, 1 - 1e-6)
            return np.log(q / (1 - q))
        return np.mean([np.log(p / (1 - p)) for p in ps], axis=0)

    def _orient(rows: np.ndarray):
        """The bout asked both ways round. Returns (logit, mirror logit)."""
        la = _p(X.iloc[rows])
        if init_score:
            la = la + _z0(rows)          # the offset is not in Booster.predict
        if not tta:
            return la, None
        lb = _p(X_mir.iloc[rows])
        if init_score:
            lb = lb + _z0(rows, mirror=True)
        return la, lb

    def _logit(rows: np.ndarray) -> np.ndarray:
        la, lb = _orient(rows)
        if lb is None:
            return la
        # the model is not exactly antisymmetric, and the half of the
        # disagreement that is noise cancels in the average
        av = 0.5 * (la - lb)
        if shr is None:
            return av
        # …and the half that does not cancel is a free per-bout confidence
        # signal that this bench currently throws away. |la + lb| is zero for a
        # perfectly antisymmetric model, so where it is large the model's own
        # answer depends on which man was typed first, and that logit deserves
        # less weight. Fitted as one interaction term on the validation slice.
        # NOT the global shrink status.md killed: an ORACLE global shrink was
        # worth 0.0005 precisely because it is one number for every bout, and
        # the diagnosis says the model is under-confident in some places and
        # over-confident in others. This is per-bout and costs nothing to
        # compute — TTA already evaluates both orientations.
        return shr[0] * av + shr[1] * av * np.abs(la + lb)

    # A model fitted to the graded target predicts E[graded], not P(win). Write
    # p for P(A wins|x) and t(x) for how decisive this matchup tends to be, and
    # the graded target's conditional mean is
    #     0.5 + (p - 0.5) * ((1 - alpha) + alpha * t(x)),
    # an affine map of p whose SLOPE moves with x. So a two-parameter logistic
    # cannot undo it — the distortion is not global. It is still monotone in p
    # (the slope is strictly positive, and t rises with the mismatch), so an
    # isotonic fit is the right inverse and a logistic one is not. Fitted on the
    # rows early stopping already used: the alternative is holding out a third
    # slice from a window the half-life has already thinned.
    shr = None
    if tta and shrink:
        from sklearn.linear_model import LogisticRegression
        la, lb = _orient(iva)
        av = 0.5 * (la - lb)
        lr = LogisticRegression(C=1e6, max_iter=2000, fit_intercept=False).fit(
            np.column_stack([av, av * np.abs(la + lb)]), B.y_all[iva].astype(int))
        shr = (float(lr.coef_[0][0]), float(lr.coef_[0][1]))

    cal = None
    if soft > 0:
        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        cal.fit(_logit(iva), B.y_all[iva].astype(float))

    def predict(rows: np.ndarray) -> np.ndarray:
        lg = _logit(rows)
        if cal is not None:
            return np.clip(cal.predict(lg), 1e-6, 1 - 1e-6)
        return np.clip(1 / (1 + np.exp(-lg)), 1e-6, 1 - 1e-6)

    predict.n_trees = float(np.mean([m.best_iteration for m in models]
                                    or [c.tree_count_ for c in cats]))
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
        # bit-for-bit the same run as base. Not a seed variant and not a
        # sanity check nobody needs: LightGBM chooses row-wise or column-wise
        # histogram building by TIMING a few iterations, so the same data on a
        # busier machine takes a different code path and sums the same floats
        # in a different order. Whatever this differs from base by is a floor
        # under every measurement on this bench, and it is not seed noise.
        "base-repeat": {},
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
        # a second model CLASS rather than a second seed. status.md lists
        # "CatBoost/LogReg" as a dead end, but as a REPLACEMENT for LightGBM;
        # nothing here has ever averaged the two, and two learners that fail
        # differently are the one ensemble that is not just variance reduction.
        "catonly": {"blend": "catonly"},
        "cat": {"blend": "cat"},
        "cat+tta": {"blend": "cat", "tta": True},
        # how members are combined, which has been logit-averaging by default
        # since the bench was written and was never measured against the mixture
        "probavg": {"avg": "prob", "seeds": 3},
        "logitavg": {"avg": "logit", "seeds": 3},
        "cat-prob": {"blend": "cat", "avg": "prob"},
        # the graded target: alpha=0 is exactly base, so this is a ladder and
        # not a different model
        "soft03": {"soft": 0.3},
        "soft06": {"soft": 0.6},
        "soft10": {"soft": 1.0},
        "soft06+tta": {"soft": 0.6, "tta": True},
        # The hyper-parameter search that found nothing swept leaves, learning
        # rate, min_data, lambda_l2 and feature_fraction — the knobs that trade
        # capacity for fit. These are a different class: they decorrelate the
        # members or smooth the leaves, and extra_trees, the first one measured,
        # is worth +0.0020 on the selection half. path_smooth in particular
        # pulls a leaf's value toward its parent in proportion to how few rows
        # reached it, which is aimed exactly at the thin-record slice where a
        # fifth of the gap to the market lives.
        "path10": {"params_over": {"path_smooth": 10.0}},
        "path50": {"params_over": {"path_smooth": 50.0}},
        "ffnode07": {"params_over": {"feature_fraction_bynode": 0.7}},
        "bin511": {"params_over": {"max_bin": 511}},
        "xt+path10": {"params_over": {"extra_trees": True, "path_smooth": 10.0}},
        "xt+tta": {"params_over": {"extra_trees": True}, "tta": True},
        "xt+ffnode": {"params_over": {"extra_trees": True,
                                      "feature_fraction_bynode": 0.7}},
        # the model's disagreement with itself under a corner swap, used as a
        # per-bout weight on its own logit. Free: TTA already computes both.
        "ttashrink": {"tta": True, "shrink": True},
        "tta-plain": {"tta": True},
        # THE LEAK. judge_ids is saved only when the scorecards were published,
        # i.e. when the bout went to a decision, so off_known and the NaN
        # pattern of every JUD column are post-bell facts. Measured on the
        # corpus: P(stoppage | off_known=1) = 0.097 against 0.743. This variant
        # is what the model would have scored knowing only what was knowable.
        "noleak": {"feats": "everyx", "drop": "jud+offknown"},
        "noleak+tta": {"feats": "everyx", "drop": "jud+offknown", "tta": True},
        "noleak+xt": {"feats": "everyx", "drop": "jud+offknown",
                      "params_over": {"extra_trees": True}},
        "noleak+xt+rot": {"feats": "everyx", "drop": "jud+offknown", "rot": True,
                          "params_over": {"extra_trees": True}},
        # extra_trees is the first regulariser on this problem that paid, and
        # the 500-trial hyper-parameter search that found nothing was run
        # WITHOUT it. A randomised split threshold under-fits each tree, so the
        # capacity optimum moves: more leaves and a lower learning rate should
        # now be affordable where they were not. This is not a re-run of that
        # search — it is the same search conditioned on a different base.
        "xt-leaves127": {"params_over": {"extra_trees": True, "num_leaves": 127}},
        "xt-leaves255": {"params_over": {"extra_trees": True, "num_leaves": 255,
                                         "learning_rate": 0.02}},
        "xt-ff06": {"params_over": {"extra_trees": True, "feature_fraction": 0.6}},
        "xt-bag07": {"params_over": {"extra_trees": True, "bagging_fraction": 0.7}},
        "xt-lr015": {"params_over": {"extra_trees": True, "learning_rate": 0.015}},
        "xt-minleaf30": {"params_over": {"extra_trees": True,
                                         "min_data_in_leaf": 30}},
        # the honest model: the post-fight judge columns out, and the card's
        # officials in their place
        # THE FORM STRIP off the event page: the last six results of each man as
        # of the night, which is recent form for the part of a career that
        # predates our crawl. Needs `--tag l6` (snapshot_extend.py).
        "x-l6": {"feats": "everyx+l6"},
        "x-l6-tta": {"feats": "everyx+l6", "tta": True},
        "x-l6-xt": {"feats": "everyx+l6", "params_over": {"extra_trees": True}},
        "x-l6-s3": {"feats": "everyx+l6", "seeds": 3},
        # the 2026-08-03 groups, one at a time on top of everyx
        "x-shr": {"feats": "everyx+shr"},
        "x-grf": {"feats": "everyx+grf"},
        "x-divr": {"feats": "everyx+divr"},
        "x-elo3": {"feats": "everyx+elo3"},
        "x-new4": {"feats": "everyx+shr+grf+divr+elo3"},
        "x-new4-s3": {"feats": "everyx+shr+grf+divr+elo3", "seeds": 3},
        "x-whr": {"feats": "everyx+whr"},
        "x-whr-s3": {"feats": "everyx+whr", "seeds": 3},
        "x-grf-whr": {"feats": "everyx+grf+whr"},
        "x-all5": {"feats": "everyx+shr+grf+divr+elo3+whr"},
        "x-all5-s3": {"feats": "everyx+shr+grf+divr+elo3+whr", "seeds": 3},
        # the same question asked of the configuration that actually ships:
        # both orientations in training, both at prediction, extra_trees
        "stack-base": {"mirror_train": True, "tta": True,
                       "params_over": {"extra_trees": True}},
        "stack-new4": {"feats": "everyx+shr+grf+divr+elo3", "mirror_train": True,
                       "tta": True, "params_over": {"extra_trees": True}},
        "stack-grf": {"feats": "everyx+grf", "mirror_train": True, "tta": True,
                      "params_over": {"extra_trees": True}},
        "stack-all5": {"feats": "everyx+shr+grf+divr+elo3+whr",
                       "mirror_train": True, "tta": True,
                       "params_over": {"extra_trees": True}},
        "stack-stk": {"feats": "everyx+shr+grf+divr+elo3", "stack": True,
                      "mirror_train": True, "tta": True,
                      "params_over": {"extra_trees": True}},
        # the walk-forward deployment number, with and without the block
        "deploy-new4": {"feats": "everyx+shr+grf+divr+elo3", "mirror_train": True,
                        "tta": True, "walk_months": 12,
                        "params_over": {"extra_trees": True}},
        "x-new5": {"feats": "everyx+shr+grf+divr+elo3+l6"},
        # the graded observation, arriving as a feature instead of as a target
        "x-stack": {"stack": True},
        "x-stack-s3": {"stack": True, "seeds": 3},
        "x-new5-stack": {"feats": "everyx+shr+grf+divr+elo3+l6", "stack": True},
        # …and the tree count chosen on the population we would bet
        "stop-prem": {"stop_on": "prem"},
        "stop-prem10": {"stop_on": "prem0.1"},
        "stop-prem-xt": {"stop_on": "prem", "params_over": {"extra_trees": True}},
        "judc": {"feats": "everyx+judc"},
        "noleak+judc": {"feats": "everyx+judc", "drop": "jud+offknown"},
        "noleak+judc+xt": {"feats": "everyx+judc", "drop": "jud+offknown",
                           "params_over": {"extra_trees": True}},
        # the deployment configuration on the honest feature set: both corner
        # orientations in training, both at prediction time, a refit every
        # twelve months, and the one regulariser that paid
        "deploy": {"mirror_train": True, "tta": True, "walk_months": 12,
                   "params_over": {"extra_trees": True}},
        "deploy-noxt": {"mirror_train": True, "tta": True, "walk_months": 12},
        "rot": {"rot": True},
        "rot+tta": {"rot": True, "tta": True},
        "xt+rot": {"rot": True, "params_over": {"extra_trees": True}},
        "xt+ttashrink": {"params_over": {"extra_trees": True}, "tta": True,
                         "shrink": True},
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
        if kw.pop("rot", False):
            c = c + [x for x in ensure_rot(BENCH) if x not in c]
        if kw.pop("stack", False):
            c = c + [x for x in ensure_stack(BENCH) if x not in c]
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
