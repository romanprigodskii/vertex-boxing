"""The one scoreboard: model against the closing line, on the same bouts.

Everything else is a knob on this. Feature sets, calibration population,
training weights and the market blend are flags, so two runs differ by exactly
what the flag says and by nothing else — the replay is cached per corpus tag.

The DEFAULTS ARE THE BEST KNOWN MODEL, not the historical ones: with --tta,
corpus 0.3346 / premium 0.2859 / -0.0260 against the close, measured 2026-08-01.
Every one of them was a measurement — no calibration because isotonic on 29k
club bouts cost 0.003 on the quoted set, a six-year half-life and five seeds
because each is worth about +0.0013, 63 leaves because a 500-trial search could
not beat it, and `everyx` because the 87 features added on 2026-08-01 are worth
+0.0025 on the confirmation half of the holdout.

  ./venv/bin/python scripts/market_eval.py --tta --blend        <- the headline
  ./venv/bin/python scripts/market_eval.py --feats everyc --drop ref --label no-ref
  ./venv/bin/python scripts/market_eval.py --tta --price open --blend --label open

--tta is on the command line rather than on by default only because it needs the
mirrored matrix, which is a second full replay the first time it is asked for.
There is no reason not to pass it: it costs one extra forward pass and it is
worth +0.0042 on the confirmation half, +0.0044 on the premium holdout and
+0.0061 on the quoted set, all with intervals clear of zero on five seeds.
--mirror trains on both orientations as well and is worth a further +0.0016, at
two and a half times the training time.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
from src import features as F  # noqa: E402

_ODDS_V2 = ROOT / "imports" / "staging" / "proboxingodds_v2.parquet"
_ODDS_V1 = ROOT / "imports" / "staging" / "proboxingodds.parquet"
ODDS = _ODDS_V2 if _ODDS_V2.exists() else _ODDS_V1
CACHE = ROOT / "imports" / "staging"
_PAREN = re.compile(r"\[.*?\]|\(.*?\)")

GROUPS = {"base": F.BASE, "record": F.RECORD, "sos2": F.SOS2, "glicko": F.GLICKO,
          "age": F.AGE, "level": F.LEVEL, "h2h": F.H2H, "dur": F.DUR,
          "form": F.FORM, "level2": F.LEVEL2, "elo2": F.ELO2, "bt": F.BT,
          "miss": F.MISS, "weigh": F.WEIGH, "score": F.SCORE, "off": F.OFF,
          "ref": F.REF, "jud": F.JUD, "card": F.CARD,
          "lvlr": F.LVLR, "lvlq": F.LVLQ, "unc": F.UNC, "res": F.RES,
          "ctx": F.CTX, "cmp": F.CMP, "thin": F.THIN, "extra": F.EXTRA}

SETS = {
    "base": F.BASE,
    "all": F.ALL,       # the 2026-07-31 feature set, with the age leak fixed
    "every": F.EVERY,   # everything computable without the scales
    "everyw": F.EVERY_W,  # …and with them — needs a snapshot that carries them
    "everys": F.EVERY_S,  # …and the judges' cards on top of that
    "everyo": F.EVERY_O,  # …and the officials who worked the bout
    "everyc": F.EVERY_C,  # …and what the saved event pages carried
    "everyx": F.EVERY_X,  # …and the levels, uncertainty, residuals and context
    # Recursive strength-of-schedule looked harmful on the 416 quoted bouts
    # (-0.0135 [-0.0253, -0.0017]) and helpful on the 75,779-bout corpus
    # holdout (+0.0023 [+0.0015, +0.0031]). The second instrument is the one
    # with the power, so it stays in — this set is kept only to show the
    # difference the choice makes.
    "best": F.BASE + F.RECORD + F.GLICKO + F.AGE + F.LEVEL,
}


def resolve(spec: str) -> list[str]:
    """'all', or groups joined by '+' — 'base+glicko+bt'. One flag covers both
    the named sets and any ablation, so a run is described by what it used."""
    if spec in SETS:
        return SETS[spec]
    cols: list[str] = []
    for part in spec.split("+"):
        src = GROUPS.get(part) or SETS.get(part)
        if src is None:
            raise SystemExit(f"unknown feature group {part!r}; "
                             f"have {', '.join(GROUPS)} or {', '.join(SETS)}")
        cols += [c for c in src if c not in cols]
    return cols


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
    """Symmetrized corpus + its features, cached — the replay is minutes.

    The cache carries the feature-set version in its name: a matrix written by
    an older definition of replay() can never be picked up by a newer one."""
    fc = CACHE / f"feats_{tag}_v{F.FEATS_VERSION}.parquet"
    dc = CACHE / f"sym_{tag}.parquet"
    if fc.exists() and dc.exists():
        return pd.read_parquet(dc), pd.read_parquet(fc)
    df = F.symmetrize(F.load(tag))
    print(f"replaying {len(df):,} bouts…", flush=True)
    feats = F.replay(df)
    df.to_parquet(dc, index=False)
    feats.to_parquet(fc, index=False)
    return df, feats


def build_mirror(tag: str, df: pd.DataFrame) -> pd.DataFrame:
    """The same matrix with the two corners exchanged.

    Not obtainable by negating columns: some are per-corner, some are ratios,
    some are invariant, and one wrong sign is a silent bug that would look like
    a feature. It is the same replay on the flipped frame — which is exact,
    because every update in replay() treats the two corners alike.
    """
    fc = CACHE / f"featsmir_{tag}_v{F.FEATS_VERSION}.parquet"
    if fc.exists():
        return pd.read_parquet(fc)
    out = df.copy()
    for x, y in F.PAIRED:
        if x in out.columns and y in out.columns:
            out[[x, y]] = out[[y, x]].values
    print(f"replaying the mirror of {len(df):,} bouts…", flush=True)
    fm = F.replay(out)
    fm.to_parquet(fc, index=False)
    return fm


def _same_man(x: str, y: str) -> bool:
    """Is this the same fighter under two spellings? Deliberately strict: a
    wrong join does not make a noisy yardstick, it makes a false one."""
    if x == y:
        return True
    tx, ty = set(x.split()), set(y.split())
    if tx <= ty or ty <= tx:
        return True
    if len(tx & ty) >= 2:
        return True
    return SequenceMatcher(None, x, y).ratio() > 0.90


def join_odds(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Pair + date, one day either side. The day is allowed to slip because a
    card that starts late local time is dated the next day by one source and
    not the other.

    Two things this used to get wrong, both measured:

    (1) The pair had to match EXACTLY on both corners, and 27% of the odds file
    failed on spelling alone — 'kenshiro teraji' against 'ken shiro'. The
    quoted set is the instrument starved of power, so throwing away a quarter
    of it to a hyphen is the expensive kind of tidy. One corner must still
    match exactly; the other is allowed to be the same man under another name.

    (2) A two-way book that implies 4.9% or 196% is not a price, it is a
    scraping error, and 4.1% of the file is like that. Proportional de-vig
    turns 37.00-vs-46.00 into a coin flip and bet_sim then settles a real
    stake at a fictitious longshot. Mirror rows are used to repair them where
    possible and they are dropped where not.
    """
    df = df.copy()
    df["na"], df["nb"] = df["a_name"].map(norm), df["b_name"].map(norm)
    df["pair"] = [f"{min(x, y)}|{max(x, y)}" if x and y else None
                  for x, y in zip(df["na"], df["nb"])]

    # best_* is the best price across the ten books on the board; close_* is
    # the worst. The old file has only the worst, so carry whichever exist.
    cars = ["close_a", "close_b", "open_a", "open_b"]
    od = pd.read_parquet(ODDS)
    if "best_a" in od.columns:
        cars += ["best_a", "best_b"]
    od["dt"] = pd.to_datetime(od["date"])
    od["na"], od["nb"] = od["a"].map(norm), od["b"].map(norm)
    od = od.dropna(subset=["na", "nb", "close_a", "close_b"])
    od = od[od["na"] != od["nb"]]
    n_raw = len(od)

    # orient every row onto the sorted pair key first, so the two mirror copies
    # of the same bout become comparable instead of merely duplicated
    flip = od["na"] > od["nb"]
    pairs = [("na", "nb"), ("close_a", "close_b"), ("open_a", "open_b")]
    if "best_a" in od.columns:
        pairs.append(("best_a", "best_b"))
    for x, y in pairs:
        od.loc[flip, [x, y]] = od.loc[flip, [y, x]].values
    od["pair"] = od["na"] + "|" + od["nb"]
    # integrity is judged on the WORST price, which must always imply more than
    # 100% — a best-of-ten-books line legitimately implies less, and filtering
    # that would throw away exactly the sharpest rows
    od["over"] = 1 / od["close_a"].astype(float) + 1 / od["close_b"].astype(float)
    od["sane"] = (od["over"] > 1.0) & (od["over"] < 1.25)
    # a sane mirror beats an insane one; otherwise keep the first
    od = (od.sort_values(["pair", "dt", "sane"], ascending=[True, True, False],
                         kind="stable")
            .drop_duplicates(subset=["pair", "dt"], keep="first"))
    bad = int((~od["sane"]).sum())
    od = od[od["sane"]]

    dbi = df.reset_index()[["index", "pair", "na", "nb", "dt"]].dropna(subset=["pair"])
    cand = dbi.merge(od[["pair", "dt", "na", *cars]], on="pair",
                     how="inner", suffixes=("", "_o"))
    cand["gap"] = (cand["dt"] - cand["dt_o"]).abs().dt.days
    cand = cand[cand["gap"] <= 1].copy()
    # which corner holds close_a. Carried as a flag, never re-derived from the
    # names downstream: an alias match has, by construction, two spellings that
    # are not equal, and a name test would silently invert its price.
    cand["swap"] = (cand["na"] != cand["na_o"]).to_numpy()
    n_exact = cand["pair"].nunique()

    # --- second pass: one corner exact, the other the same man under another
    # name. Indexed by (single name, date) so it is a lookup, not a cross join.
    got = set(zip(cand["pair"], cand["dt_o"]))
    miss = od[~pd.MultiIndex.from_arrays([od["pair"], od["dt"]]).isin(got)]
    # only the days a missing odds row could belong to — indexing all 413k
    # bouts to chase ~1,600 of them is 250x the work for the same answer
    days = {d + k for d in miss["dt"].map(pd.Timestamp.toordinal) for k in (-1, 0, 1)}
    near = dbi[dbi["dt"].map(pd.Timestamp.toordinal).isin(days)]
    by_name: dict = {}
    for t in near.itertuples(index=False):
        d0 = t.dt.toordinal()
        for nm, other, is_a in ((t.na, t.nb, True), (t.nb, t.na, False)):
            for dd in (d0 - 1, d0, d0 + 1):
                by_name.setdefault((nm, dd), []).append((t.index, other, t.dt, is_a))
    extra = []
    for o in miss.itertuples(index=False):
        d0 = o.dt.toordinal()
        for nm, other in ((o.na, o.nb), (o.nb, o.na)):
            hits = [h for h in by_name.get((nm, d0), []) if _same_man(h[1], other)]
            if len(hits) == 1:
                ix, _, dt, is_a = hits[0]
                extra.append({"index": ix, "pair": o.pair, "dt_o": o.dt,
                              "swap": (nm == o.na) != is_a,
                              "close_a": o.close_a, "close_b": o.close_b,
                              "open_a": o.open_a, "open_b": o.open_b,
                              "gap": abs((dt - o.dt).days)})
                break
    if extra:
        cand = pd.concat([cand, pd.DataFrame(extra)], ignore_index=True)

    cand = (cand.sort_values(["gap", "dt_o", "index"], kind="stable")
                .drop_duplicates("index", keep="first")
                .drop_duplicates(subset=["pair", "dt_o"], keep="first"))
    j = df.reset_index().merge(cand[["index", "swap", *cars, "gap"]],
                               on="index", how="inner")
    if verbose:
        print(f"  odds: {n_raw:,} rows → {bad:,} impossible books dropped · "
              f"{n_exact:,} matched on the exact pair · "
              f"{len(extra):,} recovered by alias")
    return j[~j["is_draw"]].reset_index(drop=True)


def devig(pa: np.ndarray, pb: np.ndarray, how: str) -> np.ndarray:
    """Turn two margin-loaded implied probabilities into A's fair probability.

    The choice is NOT cosmetic. Proportional splits the margin in proportion to
    the price, which is the most generous reading of a favourite's number;
    power and Shin both assume the book loads more margin onto the longshot,
    and both make the market look BETTER than proportional does (0.361-0.367
    against 0.376 here). Anything claimed about beating the close has to
    survive all of them.
    """
    s = pa + pb
    if how == "proportional":
        return pa / s
    if how == "additive":
        return np.clip(pa - (s - 1) / 2, 1e-4, 1 - 1e-4)
    from scipy.optimize import brentq
    out = np.empty_like(pa)
    for i, (x, z) in enumerate(zip(pa, pb)):
        if how == "power":
            try:
                k = brentq(lambda k: x ** k + z ** k - 1.0, 1.0, 400.0)
            except ValueError:
                k = 1.0
            out[i] = x ** k
        elif how == "shin":
            tot = x + z
            def f(t, x=x, z=z, tot=tot):
                return sum((np.sqrt(t * t + 4 * (1 - t) * q * q / tot) - t) / (2 * (1 - t))
                           for q in (x, z)) - 1
            try:
                t = brentq(f, 1e-9, 0.499)
            except ValueError:
                t = 1e-9
            out[i] = (np.sqrt(t * t + 4 * (1 - t) * x * x / tot) - t) / (2 * (1 - t))
        else:
            raise SystemExit(f"unknown devig: {how}")
    return np.clip(out, 1e-4, 1 - 1e-4)


def devig_slope(p: np.ndarray, y: np.ndarray) -> float:
    """How much sharpening the de-vigged price still needs to be calibrated.

    Regress the outcome on the logit of the implied probability. A slope of 1
    means the number means what it says; 1.34 means the method has flattened
    the favourite — that the price it calls 92% wins 98% of the time — and a
    model that "beats" such a number has beaten the de-vig, not the market.
    """
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return float(LogisticRegression(C=1e6, max_iter=1000)
                 .fit(z.reshape(-1, 1), y).coef_[0][0])


def bootstrap(d: np.ndarray, n: int = 4000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return tuple(np.percentile(boot, [2.5, 97.5]))


def main() -> None:  # noqa: PLR0915
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import accuracy_score, log_loss

    tag = arg("--tag", "card")
    fset = arg("--feats", "everyx")
    calib = arg("--calib", "none")       # none | all | matched | aux | regime | quoted
    weight = arg("--weight", "none")     # none | quoted
    label = arg("--label", fset)
    cols = resolve(fset)
    # leave-one-group-out: the only honest way to say which group carries the
    # gain, because groups overlap in what they explain
    drop = arg("--drop", "")
    if drop:
        gone = {c for g in drop.split("+") for c in GROUPS[g]}
        cols = [c for c in cols if c not in gone]

    df, feats = build(tag)
    feats = feats.astype("float32")
    j = join_odds(df)
    print(f"[{label}] priced bouts with an outcome: {len(j):,} "
          f"({j['dt'].min().date()} → {j['dt'].max().date()}) · "
          f"same-day {(j['gap'] == 0).mean():.0%}")

    same = ~j["swap"].to_numpy(bool)
    which = arg("--price", "close")
    ca = np.where(same, j[f"{which}_a"], j[f"{which}_b"]).astype(float)
    cb = np.where(same, j[f"{which}_b"], j[f"{which}_a"]).astype(float)
    oa = np.where(same, j["open_a"], j["open_b"]).astype(float)
    ob = np.where(same, j["open_b"], j["open_a"]).astype(float)
    if which == "best" and "best_a" not in j.columns:
        raise SystemExit("this odds file has no best_* column — re-crawl first")
    kk = np.isfinite(ca) & np.isfinite(cb) & (ca > 1) & (cb > 1)
    if not kk.all():
        j, ca, cb, oa, ob = j[kk].reset_index(drop=True), ca[kk], cb[kk], oa[kk], ob[kk]
        print(f"  {int((~kk).sum()):,} bouts dropped for a missing {which} price")
    y = (j["winner_id"].astype(str) == j["a"].astype(str)).astype(int).values
    idx = j["index"].values

    cutoff = pd.Series(j["dt"]).quantile(0.6)
    tr, te = (j["dt"] <= cutoff).values, (j["dt"] > cutoff).values
    print(f"  cutoff {cutoff.date()} · quoted train {tr.sum():,} · test {te.sum():,}")

    # The de-vig is not a matter of taste and it is not the challenger's choice
    # to make. Each method is a claim about where the bookmaker put his margin,
    # and the TRAIN slice can say which claim is true: the one whose de-vigged
    # price needs no further sharpening. Fitted before the test set is touched.
    dv = arg("--devig", "auto")
    cands = {m: devig(1 / ca, 1 / cb, m)
             for m in ("proportional", "additive", "shin", "power")}
    slopes = {m: devig_slope(v[tr], y[tr]) for m, v in cands.items()}
    if dv == "auto":
        dv = min(slopes, key=lambda m: abs(slopes[m] - 1.0))
    print("  де-виг, наклон калибровки на train: "
          + " · ".join(f"{m[:4]} {s:.3f}" + ("*" if m == dv else "")
                       for m, s in slopes.items()))
    p_mkt = cands[dv]

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
    prem_all = ((np.nan_to_num(feats["sched_rounds"].to_numpy(), nan=0) >= 8)
                & (np.minimum(feats["n_a"].to_numpy(), feats["n_b"].to_numpy()) >= 8))
    if weight == "quoted":
        w_big = w_big * np.where(prem_all[big], 3.0, 1.0)
    elif weight == "only":
        # The model loses to the price by 0.045 on quoted bouts and by nothing
        # like that on the corpus, because 90% of what it learned from is
        # four-round club boxing that no book prices. This asks the other
        # question: is the shift worth more than the 250,000 bouts it costs?
        m = prem_all[big]
        big, y_big, w_big = big[m], y_big[m], w_big[m]
    # Half of the corpus is older than the sport the market prices today. A
    # half-life says how fast a bout stops being evidence, instead of the
    # implicit "never" that training on 1950 at full weight assumes.
    hl = float(arg("--halflife", "6"))
    if hl > 0:
        yrs = ((np.datetime64(cutoff) - df["dt"].to_numpy("datetime64[D]")[big])
               / np.timedelta64(365, "D"))
        w_big = w_big * 0.5 ** (np.clip(yrs, 0, None) / hl)
    w = w_big
    cut = int(len(big) * 0.9)
    # The mirror: the same bouts entered from the other corner. A boxing match
    # has no A and no B, so the answer must not depend on which name was typed
    # first — and a tree ensemble's does. Training on both orientations makes
    # the model near-antisymmetric instead of merely trained on a symmetrised
    # sample, and --tta averages the two answers at prediction time. Worth
    # +0.0032 and +0.0022 respectively on the confirmation half, measured; the
    # mirror of a training bout stays on the training side of the split, so it
    # can never be validated against its own twin.
    ft = int(arg("--finetune", "0"))
    MIR = (build_mirror(tag, df)[cols].astype("float32")
           if {"--tta", "--mirror"} & set(sys.argv) else None)
    Xtr, ytr, wtr = X.iloc[big[:cut]], y_big[:cut], w[:cut]
    Xva, yva, wva = X.iloc[big[cut:]], y_big[cut:], w[cut:]
    if "--mirror" in sys.argv:
        if ft:
            raise SystemExit("--mirror and --finetune have not been made to "
                             "agree about the premium row mask")
        Xtr = pd.concat([Xtr, MIR.iloc[big[:cut]]], ignore_index=True)
        ytr = np.concatenate([ytr, 1.0 - ytr]); wtr = np.concatenate([wtr, wtr])
        Xva = pd.concat([Xva, MIR.iloc[big[cut:]]], ignore_index=True)
        yva = np.concatenate([yva, 1.0 - yva]); wva = np.concatenate([wva, wva])
        print(f"  зеркало: обучение на {len(Xtr):,} строках вместо {cut:,}")
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
    dva = lgb.Dataset(Xva, label=yva, weight=wva, reference=dtr)
    params = {"objective": "binary", "metric": "binary_logloss",
              "learning_rate": float(arg("--lr", "0.03")),
              "num_leaves": int(arg("--leaves", "63")),
              "min_data_in_leaf": int(arg("--minleaf", "100")),
              "feature_fraction": float(arg("--ff", "0.9")),
              "bagging_fraction": float(arg("--bf", "0.9")), "bagging_freq": 5,
              "lambda_l2": float(arg("--l2", "5.0")),
              "verbosity": -1, "seed": 42}
    if "--mono" in sys.argv:
        # A higher rating cannot make a man less likely to win. Trees do not
        # know that and will happily carve a non-monotone step out of noise in
        # a thin region; saying it out loud is free regularisation on exactly
        # the features that carry the signal.
        RATINGS = {"d_elo", "d_glicko", "d_glicko_cons", "d_bt2", "d_bt8",
                   "d_elo_mov", "d_elo_slow"}
        params["monotone_constraints"] = [1 if c in RATINGS else 0 for c in cols]
    # Seed bagging: one tree ensemble is itself a sample, and averaging a few
    # of them in logit space removes variance that early stopping cannot.
    n_seed = int(arg("--seeds", "5"))
    lgbs = []
    for k in range(n_seed):
        # not data_random_seed: that one is baked into the Dataset at
        # construction and LightGBM refuses to see it change underneath
        p = dict(params, seed=42 + k, bagging_seed=42 + k,
                 feature_fraction_seed=42 + k)
        lgbs.append(lgb.train(p, dtr, num_boost_round=4000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(150, verbose=False)]))
    # Fine-tuning: keep the 291k bouts that taught it what a boxer is, then
    # carry on boosting at a low rate over the population the market prices.
    # Cheaper than choosing between the two, and it can be measured.
    if ft:
        pm_tr = prem_all[big[:cut]]
        pm_va = prem_all[big[cut:]]
        tuned = []
        for k, m in enumerate(lgbs):
            # a fresh Dataset per seed: LightGBM bakes the seeds into the
            # handle and refuses to see them change on the next continuation
            ptr = lgb.Dataset(X.iloc[big[:cut][pm_tr]], label=y_big[:cut][pm_tr],
                              weight=w[:cut][pm_tr])
            pva = lgb.Dataset(X.iloc[big[cut:][pm_va]], label=y_big[cut:][pm_va],
                              weight=w[cut:][pm_va], reference=ptr)
            fp = dict(params, learning_rate=float(arg("--ftlr", "0.01")),
                      seed=42 + k, bagging_seed=42 + k, feature_fraction_seed=42 + k)
            tuned.append(lgb.train(fp, ptr, num_boost_round=ft, valid_sets=[pva],
                                   init_model=m,
                                   callbacks=[lgb.early_stopping(100, verbose=False)]))
        print(f"  дообучение на {int(pm_tr.sum()):,} премиальных боях: "
              + " ".join(f"{m.best_iteration}→{t.best_iteration}"
                         for m, t in zip(lgbs, tuned)))
        lgbs = tuned
    if "--refit" in sys.argv:
        # Early stopping spends the most recent tenth of the corpus on choosing
        # a tree count and then never trains on it — and under a six-year
        # half-life that tenth carries more weight than any other. Take the
        # count it found, put the validation slice back, refit on everything.
        # The count is scaled by 1/0.9 because the same number of passes now
        # has that much more data to cross.
        nr = max(int(round(np.mean([m.best_iteration for m in lgbs]) / 0.9)), 50)
        Xa, ya, wa_ = X.iloc[big], y_big, w
        if "--mirror" in sys.argv:
            Xa = pd.concat([Xa, MIR.iloc[big]], ignore_index=True)
            ya = np.concatenate([ya, 1.0 - ya]); wa_ = np.concatenate([wa_, wa_])
        dall = lgb.Dataset(Xa, label=ya, weight=wa_)
        lgbs = [lgb.train(dict(params, seed=42 + k, bagging_seed=42 + k,
                               feature_fraction_seed=42 + k),
                          dall, num_boost_round=nr) for k in range(n_seed)]
        for m in lgbs:
            m.best_iteration = nr
        print(f"  переобучение на всех {len(big):,} боях, {nr} деревьев")
    mdl = lgbs[0]
    print(f"  trained on {len(big):,} bouts · {len(cols)} features · "
          f"{'+'.join(str(m.best_iteration) for m in lgbs)} trees"
          + (f" · half-life {hl}y" if hl > 0 else ""))

    # An ensemble averaged in LOGIT space. Three models that are wrong in
    # different places: gradient boosting on leaves, gradient boosting on
    # ordered target statistics, and a plain linear model that cannot overfit
    # a rare interaction the way a tree can.
    members = [(f"lgbm{k}", (lambda m: lambda Z: m.predict(Z, num_iteration=m.best_iteration))(m))
               for k, m in enumerate(lgbs)]
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

    def _lg(Z):
        ps = [np.clip(fn(Z), 1e-6, 1 - 1e-6) for _, fn in members]
        return np.mean([np.log(p / (1 - p)) for p in ps], axis=0)

    def predict(Z):
        return 1.0 / (1.0 + np.exp(-_lg(Z)))

    def predict_rows(rows, fn=None):
        """Predict named bouts rather than a matrix, so the same bouts can be
        asked the other way round. The training data is symmetrised — a
        deterministic half of the corpus is entered B-first — but the model
        that comes out of it is not exactly antisymmetric: it will not answer
        p and 1−p to the same fight described from the two corners. Averaging
        the two logits cancels the half of that disagreement which is noise,
        and costs one extra forward pass."""
        g = fn or _lg
        lg = g(X.iloc[rows])
        if "--tta" in sys.argv:
            lg = 0.5 * (lg - g(MIR.iloc[rows]))
        return np.clip(1.0 / (1.0 + np.exp(-lg)), 1e-6, 1 - 1e-6)

    # An auxiliary model with an EARLIER cutoff. Its predictions on the window
    # between the two cutoffs are the only genuinely out-of-sample predictions
    # available before the test set — the main model has memorised everything up
    # to its own cutoff, which is why a Platt fitted on the quoted train slice
    # came out at 0.4230 against 0.4053 for no calibration at all. The blend
    # needs the same thing for λ, so it is built once and used twice.
    _aux: dict = {}

    def aux_oos():
        if _aux:
            return _aux
        c1 = cutoff - pd.DateOffset(years=int(arg("--auxback", "3")))
        akeep = (~df["is_draw"]).values & (df["dt"] <= c1).values
        abig = np.where(akeep)[0]
        acut = int(len(abig) * 0.9)
        aw = np.ones(len(abig))
        if hl > 0:
            ayrs = ((np.datetime64(c1) - df["dt"].to_numpy("datetime64[D]")[abig])
                    / np.timedelta64(365, "D"))
            aw = 0.5 ** (np.clip(ayrs, 0, None) / hl)
        adtr = lgb.Dataset(X.iloc[abig[:acut]], label=y_all[abig[:acut]],
                           weight=aw[:acut])
        adva = lgb.Dataset(X.iloc[abig[acut:]], label=y_all[abig[acut:]],
                           weight=aw[acut:], reference=adtr)
        amdl = lgb.train(params, adtr, num_boost_round=4000, valid_sets=[adva],
                         callbacks=[lgb.early_stopping(150, verbose=False)])
        win = np.where((~df["is_draw"]).values
                       & (df["dt"] > c1).values & (df["dt"] <= cutoff).values)[0]
        p = np.clip(amdl.predict(X.iloc[win], num_iteration=amdl.best_iteration),
                    1e-6, 1 - 1e-6)
        _aux.update(c1=c1, model=amdl, idx=win, p=p, n_train=len(abig))
        print(f"  вспомогательная модель до {c1.date()} ({len(abig):,} боёв), "
              f"вне обучения на {len(win):,} боях после неё")
        return _aux

    def regime(rows):
        """The uncertainty of the matchup, in the model's OWN terms.

        The band table says the failure is a scale that is wrong in opposite
        directions at the two ends: on the bouts the market calls 30-70% the
        model scores 0.768 against a coin flip's 0.693, and on 95% favourites
        it will not go past 0.90. One global slope cannot fix both. What it
        needs is a slope that depends on how identifiable the matchup is — and
        that has to be read off OUR features, never off the price, or the
        closing line stops being eval-only.
        """
        rd = np.nan_to_num(feats["rd_a"].to_numpy()[rows]
                           + feats["rd_b"].to_numpy()[rows], nan=700.0) / 100.0
        nmin = np.log1p(np.minimum(feats["n_a"].to_numpy()[rows],
                                   feats["n_b"].to_numpy()[rows]))
        sch = np.nan_to_num(feats["sched_rounds"].to_numpy()[rows], nan=6.0) / 12.0
        return np.column_stack([rd, nmin, sch])

    # Calibration. Boosting on an imbalanced target is over-confident and
    # log-loss punishes that harder than being wrong; but a calibrator fitted on
    # the four-round regional tail is being asked about a population it never
    # saw, so both the population it is fitted on and its shape are knobs.
    # `none` is one of them and is not a cop-out: an isotonic step function
    # fitted on 29k club bouts made the quoted set WORSE by 0.003, which is a
    # measurement, not a preference.
    if calib == "quoted":
        src_X, src_y = X.iloc[idx[tr]], y[tr]
    elif calib == "matched":
        m = prem_all[big[cut:]]
        src_X, src_y = X.iloc[big[cut:][m]], y_big[cut:][m]
    else:
        src_X, src_y = X.iloc[big[cut:]], y_big[cut:]

    def _ident(p, rows=None):
        return p

    if calib == "none":
        cal, n_src = _ident, 0
    elif calib in ("aux", "regime"):
        # Fitted on the auxiliary model's out-of-sample window. `aux` is one
        # slope and one intercept; `regime` lets both vary with how identifiable
        # the matchup is, which is the only shape that can be shallow on a
        # pick'em and steep on a mismatch at the same time.
        from sklearn.linear_model import LogisticRegression as _LR
        a = aux_oos()
        n_src = len(a["idx"])
        zs = np.log(a["p"] / (1 - a["p"]))

        def design(z, rows):
            if calib == "aux":
                return z.reshape(-1, 1)
            u = regime(rows)
            return np.column_stack([z, u, z[:, None] * u])

        lr = _LR(C=1.0, max_iter=2000).fit(design(zs, a["idx"]), y_all[a["idx"]])
        if calib == "regime":
            names = ["z", "rd", "log n", "sched", "z×rd", "z×log n", "z×sched"]
            print("  наклон по режиму: "
                  + " · ".join(f"{n} {c:+.3f}" for n, c in zip(names, lr.coef_[0])))

        def cal(p, rows=None, lr=lr, design=design):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
            return lr.predict_proba(design(z, rows))[:, 1]
    else:
        p_src = np.clip(predict(src_X), 1e-6, 1 - 1e-6)
        n_src = len(src_y)
        if "--platt" in sys.argv:
            # one slope and one intercept on the logit instead of a step
            # function with 29k steps: it cannot chase a bump that is not there
            from sklearn.linear_model import LogisticRegression as _LR
            z = np.log(p_src / (1 - p_src)).reshape(-1, 1)
            pl = _LR(C=1e6, max_iter=1000).fit(z, src_y)
            def cal(p, rows=None, pl=pl):
                p = np.clip(p, 1e-6, 1 - 1e-6)
                return pl.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
        else:
            iso = IsotonicRegression(out_of_bounds="clip").fit(p_src, src_y)
            def cal(p, rows=None, iso=iso):
                return iso.predict(p)
    raw_te = predict_rows(idx[te])
    p_mod = np.clip(cal(raw_te, idx[te]), 1e-4, 1 - 1e-4)
    print(f"  калибровка на {calib} (n={n_src:,}"
          f"{', platt' if '--platt' in sys.argv else ''}): "
          f"{log_loss(y[te], np.clip(raw_te, 1e-6, 1-1e-6)):.4f} → {log_loss(y[te], p_mod):.4f}")

    # The quoted test set is ~400 bouts, so it cannot see a gain of 0.005 —
    # every feature group came back "indistinguishable from zero" on it. The
    # whole post-cutoff corpus can: same model, same cutoff, 100x the bouts.
    # It says nothing about the market, and that is fine — it is the instrument
    # for "does this feature help", while the quoted set answers "do we beat
    # the price".
    post = (~df["is_draw"]).values & (df["dt"] > cutoff).values
    pidx = np.where(post)[0]
    p_corp = np.clip(cal(predict_rows(pidx), pidx), 1e-4, 1 - 1e-4)
    y_corp = y_all[pidx]
    ll_corp = log_loss(y_corp, p_corp)
    print(f"  корпусный холдаут n={len(pidx):,}: log-loss {ll_corp:.4f} · "
          f"accuracy {accuracy_score(y_corp, p_corp > .5):.3f}")
    # Third instrument. The corpus holdout has the power the quoted set lacks,
    # but four fifths of it is four-round club boxing the market never prices,
    # so a feature can win there and be irrelevant where it has to pay. This is
    # the same holdout restricted to the population the odds feed actually
    # quotes — scheduled 8 rounds or more, both men with 8 bouts behind them.
    prem = prem_all[pidx]
    ll_prem = log_loss(y_corp[prem], p_corp[prem])
    print(f"  из них «премиальные» n={int(prem.sum()):,}: log-loss {ll_prem:.4f} · "
          f"accuracy {accuracy_score(y_corp[prem], p_corp[prem] > .5):.3f}")

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
           "devig": dv, "devig_slope": slopes[dv], "price": which,
           "weight": weight, "n_test": int(te.sum()), "n_train_quoted": int(tr.sum()),
           "ll_model": float(ll_mod), "ll_market": float(ll_mkt),
           "gap": float(d.mean()), "ci": [float(lo), float(hi)],
           "ll_corpus": float(ll_corp), "n_corpus_test": int(len(pidx)),
           "ll_prem": float(ll_prem), "n_prem": int(prem.sum()),
           "halflife": hl, "seeds": n_seed,
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
        a = aux_oos()                 # one auxiliary model, used twice
        c1, amdl = a["c1"], a["model"]
        mid = ((j["dt"] > c1) & (j["dt"] <= cutoff)).values
        print(f"\n  λ подбирается на {mid.sum():,} боях после {c1.date()}, "
              f"вне обучения вспомогательной модели ({a['n_train']:,} боёв)")
        tr = mid                      # the fitting slice for the blend
        # calibrated the same way the main model is, so λ mixes two logits on
        # one scale instead of also absorbing a scale difference
        p_tr = np.clip(cal(amdl.predict(X.iloc[idx[tr]],
                                        num_iteration=amdl.best_iteration),
                           idx[tr]), 1e-4, 1 - 1e-4)
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

        if "--fade" in sys.argv:
            # A constant λ says "trust the model this much, always". The test
            # set says otherwise: the model is calibrated overall (slope 0.94)
            # and falls apart exactly where the price disagrees with it — on the
            # bouts the market calls even, its 0.82 predictions land at 0.535,
            # a slope of 0.24. So let the weight FADE as the disagreement grows:
            # λ(d) = λ0 / (1 + c·|z_model − z_market|). Two parameters instead of
            # one, fitted on the same out-of-sample window, and c = 0 recovers
            # the constant blend exactly — so it cannot do worse by construction
            # on the fitting slice, only on the test set, which is the point.
            dtr_, dte_ = np.abs(ltr_m - ltr_k), np.abs(lte_m - lte_k)
            best = (1e9, lam, 0.0)
            for g in np.linspace(0, 1, 51):
                for c in np.concatenate([[0.0], np.geomspace(0.01, 3.0, 40)]):
                    w = g / (1 + c * dtr_)
                    q = 1 / (1 + np.exp(-(w * ltr_m + (1 - w) * ltr_k)))
                    L = log_loss(y[tr], q)
                    if L < best[0]:
                        best = (L, float(g), float(c))
            _, g0, c0 = best
            w_te = g0 / (1 + c0 * dte_)
            p_bl = np.clip(1 / (1 + np.exp(-(w_te * lte_m + (1 - w_te) * lte_k))),
                           1e-6, 1 - 1e-6)
            cm, ck = float(np.mean(w_te)), float(1 - np.mean(w_te))
            print(f"\n  ЗАТУХАЮЩИЙ БЛЕНД  λ0 {g0:.2f} · c {c0:.3f} · "
                  f"средняя λ на тесте {cm:.3f} "
                  f"(при полном согласии {g0:.2f}, при |Δlogit|=2 "
                  f"{g0 / (1 + c0 * 2):.3f})")
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
             # devigged probability, but a bet is settled at the real number.
             # oa/ob are the OPEN, so a bet struck early can be marked to the
             # close and the line movement scored as CLV.
             ca=ca[te], cb=cb[te], oa=oa[te], ob=ob[te], price=which,
             p_blend=(p_bl if "--blend" in sys.argv else p_mod),
             p_corp=p_corp, y_corp=y_corp, key_corp=pidx, prem_corp=prem)


if __name__ == "__main__":
    main()
