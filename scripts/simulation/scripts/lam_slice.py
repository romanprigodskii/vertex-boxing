"""Where does the model still know something the price does not?

The blend answers that with ONE number: λ ≈ 0.16 of the model's logit, the rest
the market's. That is an average over a population the project has just shown to
be wildly heterogeneous — the same test set gives +0.0043 of CLV on club cards
and +0.0183 on twelve-rounders. An average λ says nothing about where the 0.16
comes from.

This fits λ SEPARATELY inside slices that are all knowable before the bell, and
it is a diagnostic, not a model. λ→0 in a slice means our features add nothing
there and no amount of feature work on that population will pay; a high λ means
the price is missing what we have, and that is where another column is worth
writing. `--bandblend` asked the same question of the PRICE axis and lost as a
model; the point here is not to deploy the per-slice weights but to read them.

Two things it must not do, both of which this project has done before:
  * fit λ on bouts the main model was trained on. Its predictions there are
    in-sample and λ comes out near 0.85 because the model remembers them. So λ
    is fitted against an AUXILIARY model trained to an earlier cutoff, exactly
    as market_eval does.
  * report a slice's λ without its n. Four parameters carved out of 2,472 bouts
    is what killed the band blend, and the same arithmetic applies here.

  python3 scripts/lam_slice.py --tag l6 --feats everyz --json results/lambda_by_slice.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(ROOT / "scripts" / "simulation" / "scripts"))

import lab  # noqa: E402
import market_eval as ME  # noqa: E402


def lg(p):
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(np.where(y == 1, np.log(p), np.log1p(-p))))


def fit_lam(lm, lk, y) -> tuple[float, float]:
    """The one-parameter mix, on a grid. Returns (λ, its log-loss)."""
    grid = np.linspace(0, 1, 101)
    losses = [ll(1 / (1 + np.exp(-(g * lm + (1 - g) * lk))), y) for g in grid]
    k = int(np.argmin(losses))
    return float(grid[k]), float(losses[k])


def slices(B, rows: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Every cut is a fact about the bout, fixed before the first bell, and
    none of them mentions the price."""
    f = B.feats
    sch = np.nan_to_num(f["sched_rounds"].to_numpy()[rows], nan=0)
    tit = np.nan_to_num(f["title_lvl"].to_numpy()[rows], nan=0)
    nmin = np.minimum(f["n_a"].to_numpy()[rows], f["n_b"].to_numpy()[rows])
    same = f["same_ctry"].to_numpy()[rows]
    out = [
        ("distance ≤6", sch <= 6), ("distance 8", sch == 8),
        ("distance 10", sch == 10), ("distance 12", sch >= 12),
        ("no belt", tit == 0), ("regional/national belt", (tit >= 1) & (tit <= 2)),
        ("continental/world belt", tit >= 3),
        ("thinner man <3 bouts", nmin < 3), ("3–8", (nmin >= 3) & (nmin < 8)),
        ("8–15", (nmin >= 8) & (nmin < 15)), ("15+", nmin >= 15),
        ("same country", same == 1), ("different countries", same == 0),
    ]
    return out


def main() -> None:
    tag = ME.arg("--tag", "l6")
    fset = ME.arg("--feats", "everyx")
    back = int(ME.arg("--auxback", "3"))
    B = lab.Bench(tag)
    lab.BENCH = B
    cols = ME.resolve(fset)

    c1 = B.cutoff - pd.DateOffset(years=back)
    print(f"auxiliary model to {c1.date()}, λ fitted on quoted bouts after it "
          f"and up to the cutoff {B.cutoff.date()}", flush=True)
    aux = lab.fit(B, cols, c1, seeds=1, tta=True)
    main_ = lab.fit(B, cols, B.cutoff, seeds=1, tta=True)

    jdt = pd.Series(B.dt[B.jidx])
    mid = ((jdt > c1) & (jdt <= B.cutoff)).to_numpy()
    te = B.qte
    p_mid = aux(B.jidx[mid])
    p_te = main_(B.jidx[te])
    y_mid, y_te = B.jy[mid], B.jy[te]
    lm_mid, lk_mid = lg(p_mid), lg(B.p_mkt[mid])
    lm_te, lk_te = lg(p_te), lg(B.p_mkt[te])

    lam0, _ = fit_lam(lm_mid, lk_mid, y_mid)
    p_bl = 1 / (1 + np.exp(-(lam0 * lm_te + (1 - lam0) * lk_te)))
    res: dict = {"tag": tag, "feats": fset, "aux_cutoff": str(c1.date()),
                 "cutoff": str(B.cutoff.date()), "global_lambda": lam0,
                 "n_fit": int(mid.sum()), "n_test": int(te.sum()),
                 "ll_market": ll(B.p_mkt[te], y_te), "ll_model": ll(p_te, y_te),
                 "ll_blend": ll(p_bl, y_te), "slices": []}
    print(f"\nglobal λ {lam0:.2f} on {int(mid.sum()):,} fitting bouts · "
          f"test {int(te.sum()):,}: market {ll(B.p_mkt[te], y_te):.4f} · "
          f"model {ll(p_te, y_te):.4f} · blend {ll(p_bl, y_te):.4f}")

    print(f"\n{'slice':24s} {'n fit':>6s} {'n test':>6s} {'λ':>5s} "
          f"{'market':>8s} {'model':>8s} {'blend@λ':>8s} {'gain':>8s}")
    mid_rows, te_rows = B.jidx[mid], B.jidx[te]
    for (name, m_mid), (_, m_te) in zip(slices(B, mid_rows), slices(B, te_rows)):
        if m_mid.sum() < 60 or m_te.sum() < 60:
            print(f"{name:24s} {int(m_mid.sum()):6,} {int(m_te.sum()):6,} "
                  f"{'—':>5s}   too few to fit")
            continue
        lam, _ = fit_lam(lm_mid[m_mid], lk_mid[m_mid], y_mid[m_mid])
        bl = 1 / (1 + np.exp(-(lam * lm_te[m_te] + (1 - lam) * lk_te[m_te])))
        mk = ll(B.p_mkt[te][m_te], y_te[m_te])
        md = ll(p_te[m_te], y_te[m_te])
        print(f"{name:24s} {int(m_mid.sum()):6,} {int(m_te.sum()):6,} "
              f"{lam:5.2f} {mk:8.4f} {md:8.4f} {ll(bl, y_te[m_te]):8.4f} "
              f"{mk - ll(bl, y_te[m_te]):+8.4f}")
        res["slices"].append({"slice": name, "n_fit": int(m_mid.sum()),
                              "n_test": int(m_te.sum()), "lambda": lam,
                              "ll_market": mk, "ll_model": md,
                              "ll_blend": ll(bl, y_te[m_te])})

    print("\nRead the λ column, not the gain column: the gain is measured on a "
          "few hundred bouts and moves by more than it is worth. λ is fitted on "
          "the other window and says how much of the model the price does not "
          "already contain, which is the question that decides where to spend "
          "the next feature.")
    if "--json" in sys.argv:
        dst = Path(ME.arg("--json", ""))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
