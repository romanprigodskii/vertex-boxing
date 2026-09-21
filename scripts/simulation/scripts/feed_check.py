"""Why the merged price feed says the model is level with the market, and why that is wrong.

On the merge of every price source (odds_merge.py: the ProBoxingOdds re-crawl,
OddsPortal, Oddschecker) the scoreboard reads a gap of −0.002 against −0.021 on
ProBoxingOdds alone. Nothing about the model changed; the priced set did. This
splits the merged run into the bouts both feeds price and the bouts only the
merge adds, and attributes every bout whose price changed to the source that
supplied the new one.

What it found on 2026-09-21: on the bouts where the merge took OddsPortal's price
the favourite is often on the wrong corner — the outcome sides with the original
feed — and on bouts only OddsPortal prices, the "market" scores about 1.0 nats,
worse than a coin. An orientation fault in the OddsPortal ingestion, not an edge.
The merged feed is not used for any published number until that is fixed.

  python3 scripts/feed_check.py final-close final-close-allodds --json results/diagnostics/merged_feed.json
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
import market_eval as ME  # noqa: E402

CACHE = ROOT / "imports" / "staging"


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    # positional labels; skip every flag and the value that follows it
    args, skip = [], False
    for x in sys.argv[1:]:
        if skip:
            skip = False
        elif x.startswith("--"):
            skip = x in ("--tag", "--json")
        else:
            args.append(x)
    base_label = args[0] if args else "final-close"
    merged_label = args[1] if len(args) > 1 else "final-close-allodds"
    tag = ME.arg("--tag", "l6")
    a = np.load(CACHE / "preds" / f"{base_label}.npz")
    b = np.load(CACHE / "preds" / f"{merged_label}.npz")
    A = pd.DataFrame({"key": a["key"], "caA": a["ca"], "cbA": a["cb"], "mA": a["p_mkt"],
                      "pA": a["p"], "y": a["y"]})
    B = pd.DataFrame({"key": b["key"], "caB": b["ca"], "cbB": b["cb"], "mB": b["p_mkt"],
                      "pB": b["p"], "yB": b["y"]})
    both = A.merge(B, on="key")
    assert (both.y == both.yB).all(), "the two runs disagree on who won"
    added = B[~B.key.isin(A.key)].copy()
    changed = both[~(np.isclose(both.caA, both.caB) & np.isclose(both.cbA, both.cbB))].copy()
    changed["flip"] = (changed.mA > 0.5) != (changed.mB > 0.5)

    # which source the merge took each new price from: same day ±1, either name
    sym = pd.read_parquet(CACHE / f"sym_{tag}.parquet", columns=["dt", "a_name", "b_name"])
    sym["dt"] = pd.to_datetime(sym["dt"])
    od = pd.read_parquet(CACHE / "odds_all.parquet")
    od["when"] = pd.to_datetime(od["date"])
    od["na"], od["nb"] = od["a"].map(ME.norm), od["b"].map(ME.norm)

    def source_of(k):
        r = sym.iloc[k]
        names = [ME.norm(r["a_name"]), ME.norm(r["b_name"])]
        m = od[((od["when"] - r["dt"]).abs().dt.days <= 1)
               & (od["na"].isin(names) | od["nb"].isin(names))]
        return "+".join(sorted(set(m["source"]))) or "unknown"

    changed["src"] = changed.key.map(source_of)
    added["src"] = added.key.map(source_of)
    g = ll(B.mB, B.yB) - ll(B.pB, B.yB)
    inB = B.key.isin(A.key).to_numpy()
    out = {"base": base_label, "merged": merged_label,
           "n_base": int(len(A)), "n_merged": int(len(B)), "n_shared": int(len(both)),
           "n_added": int(len(added)),
           "gap_merged": float(g.mean()),
           "gap_contribution_shared": float(g[inB].sum() / len(B)),
           "gap_contribution_added": float(g[~inB].sum() / len(B)),
           "shared": {"market_base": float(ll(both.mA, both.y).mean()),
                      "market_merged": float(ll(both.mB, both.y).mean()),
                      "identical_close_share": float(1 - len(changed) / len(both))},
           "changed_by_source": {}, "added_by_source": {}}
    print(f"{len(both):,} bouts in both runs, {len(added):,} only in the merged one")
    print(f"merged gap {out['gap_merged']:+.4f} = shared {out['gap_contribution_shared']:+.4f} "
          f"+ added {out['gap_contribution_added']:+.4f}")
    print(f"on shared bouts the market scores {out['shared']['market_base']:.4f} on the "
          f"original feed and {out['shared']['market_merged']:.4f} on the merge\n")
    print("bouts whose closing price the merge changed, by the source of the new price")
    for s, x in changed.groupby("src"):
        rec = {"n": int(len(x)), "favourite_flipped": float(x.flip.mean()),
               "market_base": float(ll(x.mA, x.y).mean()),
               "market_merged": float(ll(x.mB, x.y).mean())}
        if x.flip.any():
            f = x[x.flip]
            rec["base_favourite_won_when_flipped"] = float(((f.mA > 0.5) == (f.y == 1)).mean())
        out["changed_by_source"][s] = rec
        print(f"  {s:28s} {rec['n']:4d}  favourite flipped {rec['favourite_flipped']:4.0%}  "
              f"market {rec['market_base']:.3f} → {rec['market_merged']:.3f}")
    print("bouts only the merge prices, by source")
    for s, x in added.groupby("src"):
        rec = {"n": int(len(x)), "market": float(ll(x.mB, x.yB).mean()),
               "model": float(ll(x.pB, x.yB).mean())}
        out["added_by_source"][s] = rec
        print(f"  {s:28s} {rec['n']:4d}  market {rec['market']:.3f}  model {rec['model']:.3f}")
    if "--json" in sys.argv:
        dst = Path(ME.arg("--json", ""))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
