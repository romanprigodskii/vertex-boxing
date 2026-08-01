"""The mirrored matrix must be the mirror, to the last bit.

Test-time symmetrisation averages the model's answer to a bout with its answer
to the same bout entered from the other corner. That is only a variance
reduction if the second matrix really is the first one with the corners
exchanged — and it is produced by a second replay, not by negating columns, so
nothing guarantees it except this test. A feature that failed to flip would not
crash anything; it would quietly feed the model a contradiction and look like a
result.

Every column must fall into exactly one class:
  anti  a difference, so it negates                       d_elo, d_glicko
  inv   invariant under the swap                          elo_min, sched_rounds
  prob  a probability of A, so it becomes 1 − p           glicko_e
  swap  half of an a/b pair, so it trades with its twin   n_a ↔ n_b

The classes are DERIVED from the names, not listed by hand, so a column added
tomorrow is tested tomorrow.

  ./venv/bin/python scripts/mirror_check.py card
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
from src import features as F  # noqa: E402

CACHE = ROOT / "imports" / "staging"
TOL = 1e-6

# the handful whose class cannot be read off the name
PROB = {"glicko_e", "h2h_score"}
def classify(c: str, cols: set[str]) -> tuple[str, str | None]:
    if c in PROB:
        return "prob", None
    for suf, other in (("_a", "_b"), ("_b", "_a")):
        if c.endswith(suf) and c[:-2] + other in cols:
            return "swap", c[:-2] + other
    for pre, other in (("a_", "b_"), ("b_", "a_")):
        if c.startswith(pre) and other + c[2:] in cols:
            return "swap", other + c[2:]
    if c.startswith("d_") or c.endswith("_jud") or c.endswith("_ref"):
        return "anti", None
    return "inv", None


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "card"
    v = F.FEATS_VERSION
    o = pd.read_parquet(CACHE / f"feats_{tag}_v{v}.parquet")
    m = pd.read_parquet(CACHE / f"featsmir_{tag}_v{v}.parquet")
    cols = set(o.columns)
    assert cols == set(m.columns), "the two matrices do not have the same columns"
    print(f"{len(o):,} bouts · {len(cols)} columns · feature version {v}\n")
    bad, counts = [], {"anti": 0, "inv": 0, "prob": 0, "swap": 0}
    for c in o.columns:
        kind, twin = classify(c, cols)
        counts[kind] += 1
        a = o[c].to_numpy(float)
        b = m[twin if twin else c].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        # missingness must mirror too: a NaN on one side and a number on the
        # other is the same defect in a quieter form
        nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
        if ok.sum() == 0:
            continue
        if kind == "anti":
            err = np.abs(a[ok] + b[ok])
        elif kind == "prob":
            err = np.abs(a[ok] + b[ok] - 1.0)
        else:
            err = np.abs(a[ok] - b[ok])
        scale = max(np.abs(a[ok]).max(), 1.0)
        rel = err.max() / scale
        if rel > TOL or nan_mismatch:
            bad.append((c, kind, rel, nan_mismatch))
    print("  by class: " + " · ".join(f"{k} {n}" for k, n in counts.items()))
    if bad:
        print(f"\n  {len(bad)} COLUMNS DO NOT MIRROR:")
        for c, kind, rel, nm in sorted(bad, key=lambda r: -r[2]):
            print(f"    {c:20s} as {kind:5s} rel-err {rel:.2e} · "
                  f"{nm:,} rows missing on one side only")
        raise SystemExit(1)
    print("\n  every column mirrors  ✓")


if __name__ == "__main__":
    main()
