"""Two auxiliary models, whose PREDICTIONS become features of the main one.

status.md killed the graded target and explained exactly why: the judges hand
out a number on 144,337 bouts and the referee's stopping round grades 267,000
more, none of it reaches the objective, and training on the grade instead of the
bit makes the model predict

    0.5 + (p − 0.5)·((1 − α) + α·t(x))

where t(x) is how decisive a matchup of this kind usually is. The slope depends
on the bout, so no one-dimensional inverse recovers p, and the information is
not worth the distortion. That is an argument against the graded TARGET. It is
not an argument against the graded OBSERVATION.

So the grade goes in through the front door instead: fit a model to predict it,
and hand the prediction to the main model as a column. The main model keeps its
0/1 label and its calibrated loss; what it gains is an estimate of how this
particular fight is likely to be decided, which is the thing every hand-built
proxy in the feature file gropes at. `p_stop_hat` today is the average of four
raw rates with no interaction and no shrinkage.

Both targets are POST-BELL, so the predictions have to be point-in-time or the
column is a leak wearing a hat:

  * a block is predicted by a model trained ONLY on bouts dated before it, in
    expanding windows. No row ever contributes to the model that scores it, and
    no row is ever scored by a model that has seen the future.
  * the earliest block has nothing to train on and stays NaN, which is honest
    and which LightGBM already knows how to route.

Both are made exactly symmetric here rather than hoped to be: P(stoppage) is
averaged over the two corner orders and the dominance estimate is halved
antisymmetrically, so the mirrored matrix can be written down instead of
re-predicted, and mirror_check still has something to check.

  ./venv/bin/python scripts/stack.py --tag l6
"""

from __future__ import annotations

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
COLS = ["stack_pstop", "stack_dom"]
# expanding-window blocks. Three years is a compromise: finer means more fits
# for the same data, coarser means the last block is predicted by a model up to
# three years stale, which is what walk-forward already measures at 12 months.
BLOCK_YEARS = 3
FIRST = 1975


def targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(stoppage flag, dominance, which rows have a usable grade)."""
    meth = df["method"].astype(str).str.lower().to_numpy()
    stop = np.isin(meth, list(F.STOP)).astype(float)
    known = np.isin(meth, list(F.STOP) + ["ud", "sd", "md", "pts",
                                          "technical_decision"])
    sa = np.where(df["is_draw"].to_numpy(), 0.5,
                  (df["winner_id"].astype(str).to_numpy()
                   == df["a"].astype(str).to_numpy()).astype(float))
    a_sc = pd.to_numeric(df.get("a_score"), errors="coerce").to_numpy(float) \
        if "a_score" in df.columns else np.full(len(df), np.nan)
    b_sc = pd.to_numeric(df.get("b_score"), errors="coerce").to_numpy(float) \
        if "b_score" in df.columns else np.full(len(df), np.nan)
    rf = pd.to_numeric(df["round_finished"], errors="coerce").to_numpy(float)
    sch = pd.to_numeric(df["sched"], errors="coerce").to_numpy(float)
    dom = np.array([F.dominance(sa[i], a_sc[i], b_sc[i], meth[i], rf[i], sch[i])
                    for i in range(len(df))], dtype=float)
    return stop, dom, known


def main() -> None:
    import lightgbm as lgb
    tag = ME.arg("--tag", "l6")
    fset = ME.arg("--feats", "everyx")
    df, feats = ME.build(tag)
    mir = ME.build_mirror(tag, df)
    cols = [c for c in ME.resolve(fset) if c in feats.columns]
    X = feats[cols].astype("float32")
    Xm = mir[cols].astype("float32")
    stop, dom, known = targets(df)
    dt = df["dt"].to_numpy("datetime64[D]")
    years = dt.astype("datetime64[Y]").astype(int) + 1970

    out = {c: np.full(len(df), np.nan) for c in COLS}
    params = {"objective": "binary", "metric": "binary_logloss",
              "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
              "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
              "lambda_l2": 5.0, "verbosity": -1, "seed": 42, "num_threads": 8,
              "force_row_wise": True, "deterministic": True}
    reg = dict(params, objective="regression", metric="l2")

    edges = list(range(FIRST, int(years.max()) + BLOCK_YEARS, BLOCK_YEARS))
    for lo, hi in zip(edges[:-1], edges[1:]):
        past = (years < lo) & known
        blk = (years >= lo) & (years < hi)
        if blk.sum() == 0 or past.sum() < 5000:
            continue
        t0 = time.time()
        ip, ib = np.where(past)[0], np.where(blk)[0]
        # half-life on the training weight, same six years the main model uses
        w = 0.5 ** (np.clip((lo - years[ip]), 0, None) / 6.0)
        m1 = lgb.train(params, lgb.Dataset(X.iloc[ip], label=stop[ip], weight=w,
                                           params=params), num_boost_round=300)
        m2 = lgb.train(reg, lgb.Dataset(X.iloc[ip], label=dom[ip], weight=w,
                                        params=reg), num_boost_round=300)
        # exactly symmetric by construction, not by hope
        out["stack_pstop"][ib] = 0.5 * (m1.predict(X.iloc[ib])
                                        + m1.predict(Xm.iloc[ib]))
        out["stack_dom"][ib] = 0.5 * (m2.predict(X.iloc[ib])
                                      - m2.predict(Xm.iloc[ib]))
        print(f"  {lo}–{hi - 1}: trained on {len(ip):,} · scored {len(ib):,} "
              f"· {time.time() - t0:.0f}s", flush=True)

    o = pd.DataFrame(out)
    path = CACHE / f"stack_{tag}{F.tune_tag()}_v{F.FEATS_VERSION}.parquet"
    o.to_parquet(path, index=False)
    ok = o["stack_pstop"].notna()
    print(f"{len(o):,} rows → {path.name} · filled on {ok.mean():.1%}")
    print(f"  P(stoppage) predicted mean {o.loc[ok, 'stack_pstop'].mean():.3f} "
          f"against actual {stop[ok.to_numpy() & known].mean():.3f}")


if __name__ == "__main__":
    main()
