# Vertex Boxing

A fundamentals model of professional boxing, scored against the bookmaker's
closing line. It was built to test one thesis: that regional and club boxing
lines are soft enough for a model that reads records and ratings to find room.
**The data says the opposite**, and this repository is the record of how that was
measured.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/clv_by_distance-dark.svg">
  <img alt="Closing-line value by scheduled distance: +0.0044 on 4–6 rounders, +0.0075 on 8, +0.0093 on 10, +0.0180 on 12, each with a 95% interval" src="docs/figures/clv_by_distance-light.svg" width="720">
</picture>

| | |
|---|---|
| model against the closing line, 3,288 priced bouts | **−0.0211** [−0.0322, −0.0100] nats |
| the same, retrained yearly as a deployment would be | −0.0198 [−0.0311, −0.0086] |
| the model blended into the closing price, λ = 0.17 | **+0.0043** [+0.0024, +0.0062] |
| model against the *opening* line | −0.0089 [−0.0206, +0.0026] |
| closing-line value, 4–6 rounders → 12-rounders | +0.0044 → **+0.0180** |
| the level rule on a window it was never chosen on | upper tier ×2.0 the CLV of the lower: +0.0276 vs +0.0136 |
| return at the closing price, upper tier | −0.6% (proportional) to **−6.1%** (power de-vig) |
| a pre-registered random search, 200 configurations | nothing survives on unseen bouts |
| a post-bell leak caught in the project's own data | P(stoppage) 0.097 vs 0.743 |
| real bets placed | 0 |

On its own the model loses to the close. Blended into the price, it improves the
price, so it knows something the price does not. And it knows it at the **top**
of the sport, not the bottom: closing-line value rises with the scheduled
distance, and three independent markers of level agree. The margin paid at the
open is larger than the movement the model catches, so none of this is money.

**Read [`docs/REPORT.md`](docs/REPORT.md)**: the question, the data, the model,
the protocol, every result with its interval, the leak, and the dead ends.

## What is here

```
docs/REPORT.md               the final report — start here
docs/model.md                the lab notebook, in English (every pass, every dead end)
docs/status.md               the lab notebook, in Russian (the 2026-08-04 state in detail)
docs/odds.md                 where historical boxing prices exist, and where they do not
scripts/simulation/
  src/features.py            the point-in-time replay: 231 features, both orientations
  scripts/market_eval.py     the scoreboard: model vs the closing line, and the blend
  scripts/lab.py             the bench: many variants, one paired verdict each
  scripts/regional.py        the level cut
  scripts/rule_oos.py        the level rule on an era it never saw
  scripts/clv_money.py       closing-line value into money, margin included
  scripts/leak_check.py      does any feature say how the fight ended?
  scripts/mirror_check.py    does every feature mirror when the corners swap?
  reproduce.sh               every number in the report, one command
  results/                   what reproduce.sh wrote — the files the report cites
```

## What is not here, and why

**The data.** The corpus of 413,279 bouts was assembled from BoxRec, Wikipedia
and Wikidata, and the prices come from ProBoxingOdds. BoxRec's terms forbid
redistributing data derived from it, and the prices are a third party's.
`results/data_manifest.json` holds the SHA-256 of every input file, so a copy
shown to a reviewer can be checked against what the results were computed from.
**The data-collection code** is not published for the same reason.

## Reproducing

With the data in `imports/staging/` (Python 3.12, `pip install -r
scripts/simulation/requirements.txt`):

```bash
cd scripts/simulation
./reproduce.sh              # about 2.5 hours on an 8-core M3
./reproduce.sh level_cut    # one step
```

The bench is deterministic: LightGBM is pinned to one histogram code path, the
replay sorts every set it sums over, and every bootstrap is seeded. Re-run on
2026-09-21, the scoreboard published on 2026-08-04 came back identical in every
field to 16 significant digits. The library versions are pinned in
`requirements.txt` and recorded in `results/environment.json`.

## Status

Concluded as research, September 2026. The one experiment that could still
change the money answer, and that history cannot run, is a forward test:
publishing predictions before the bell and grading them against prices that were
live at the time.

Code: MIT. Part of [prigodskii.dev](https://prigodskii.dev); the MMA system this
was ported from is [Vertex MMA](https://github.com/romanprigodskii/vertex-mma).
