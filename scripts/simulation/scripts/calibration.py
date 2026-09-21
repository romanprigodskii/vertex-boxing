"""Is the gap to the close a calibration problem? Two numbers say no.

The first is the calibration slope on each instrument: regress the outcome on
the logit of the model's probability. 1 means the number means what it says;
below 1 the model is over-confident, above it timid. The second is an oracle: the
single factor on the logit that minimises log-loss ON THE TEST SET ITSELF, which
no honest model can have. What that oracle buys is an upper bound on anything a
calibrator could ever recover — and if it is a few ten-thousandths, the gap is
missing information, not misplaced confidence.

Reads one saved scoreboard run; trains nothing.

  python3 scripts/calibration.py final-close --json results/calibration.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "imports" / "staging"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(np.where(y == 1, np.log(p), np.log1p(-p))))


def slope(p, y) -> float:
    return float(LogisticRegression(C=1e6, max_iter=1000)
                 .fit(logit(p).reshape(-1, 1), y).coef_[0][0])


def oracle(p, y) -> tuple[float, float]:
    """The best single logit multiplier chosen on these very bouts, and what
    it buys over leaving the model alone."""
    z = logit(p)
    grid = np.linspace(0.5, 1.5, 201)
    losses = [ll(1 / (1 + np.exp(-s * z)), y) for s in grid]
    k = int(np.argmin(losses))
    return float(grid[k]), ll(p, y) - losses[k]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    label = args[0] if args else "final-close"
    d = np.load(CACHE / "preds" / f"{label}.npz", allow_pickle=True)
    inst = {"corpus": (d["p_corp"], d["y_corp"]),
            "premium": (d["p_corp"][d["prem_corp"]], d["y_corp"][d["prem_corp"]]),
            "quoted": (d["p"], d["y"])}
    out = {"label": label}
    print(f"{label}")
    print(f"{'instrument':10s} {'n':>7s} {'slope':>7s} {'oracle factor':>14s} "
          f"{'oracle buys':>12s}")
    for name, (p, y) in inst.items():
        s = slope(p, y)
        f, gain = oracle(p, y)
        out[name] = {"n": int(len(y)), "calibration_slope": s,
                     "oracle_factor": f, "oracle_gain": gain}
        print(f"{name:10s} {len(y):7,} {s:7.3f} {f:14.3f} {gain:+12.4f}")
    if "--json" in sys.argv:
        dst = Path(sys.argv[sys.argv.index("--json") + 1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
