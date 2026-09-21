"""The margin in each reading of the board, on the priced test bouts.

The scoreboard compares the model with three readings of the same market: the
worst price of the ten books at the close, the best price on the board, and the
open. Each carries a different margin, and the report quotes it beside each
comparison — this is where that number comes from.

  python3 scripts/board_margin.py --json results/board_margins.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "imports" / "staging"


def main() -> None:
    out = {}
    for price in ("close", "best", "open"):
        d = np.load(CACHE / "preds" / f"final-{price}.npz")
        ca, cb = d["ca"], d["cb"]
        k = np.isfinite(ca) & np.isfinite(cb) & (ca > 1) & (cb > 1)
        ov = 1 / ca[k] + 1 / cb[k]
        out[price] = {"bouts": int(k.sum()), "overround_mean": float(ov.mean()),
                      "overround_median": float(np.median(ov)),
                      "margin_mean": float(ov.mean() - 1)}
        print(f"{price:5s} {int(k.sum()):,} bouts · overround {ov.mean():.4f} "
              f"(median {np.median(ov):.4f}) · margin {ov.mean() - 1:.1%}")
    if "--json" in sys.argv:
        dst = Path(sys.argv[sys.argv.index("--json") + 1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
