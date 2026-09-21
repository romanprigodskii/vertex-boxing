# Vertex Boxing — the model and the bench

What the model is, how it is scored, and what it found:
[`docs/REPORT.md`](../../docs/REPORT.md). This file is the map of the code.

## The model

`src/features.py` is the whole of it. `load(tag)` reads a frozen corpus snapshot,
`symmetrize()` flips a deterministic half of the bouts so that corner A is not
the winner 87% of the time, and `replay()` walks the corpus in date order and
writes, for every bout, what was known about both men before its first bell:
ratings, records, strength of schedule, activity, the level of the bout, the
officials on the card, and the comparability block. Feature sets are named in
`market_eval.py` (`everyz` is the final one, 231 columns). The matrices are
cached per snapshot tag and feature version, so a matrix built by an older
`replay()` can never be picked up by a newer one.

The model itself is a LightGBM binary classifier trained on both orientations
of every bout, averaged over both at prediction time, with extremely randomised
trees, a six-year half-life on training weights and three seeds.

## The scripts that produce the published numbers

| script | what it answers |
|---|---|
| `market_eval.py` | the scoreboard: the model and the blend against one reading of the market's price |
| `lab.py` | the bench: many variants on one data load, each against a base by a paired bootstrap; also the yearly walk-forward deployment |
| `regional.py` | closing-line value and return cut by the level of the bout |
| `rule_oos.py` | the level rule applied, unedited, to an era it was never chosen on |
| `clv_money.py` | closing-line value turned into money, margin included, under both de-vig methods |
| `lam_slice.py` | the blend weight fitted inside slices known before the bell |
| `calibration.py`, `where.py`, `polymarket_eval.py` | calibration, where the gap sits, and a price with no margin |
| `board_margin.py`, `feed_check.py` | the margin in each reading of the board; why the merged price feed is not used |
| `leak_check.py`, `mirror_check.py` | correctness: no feature may say how the fight ended; every feature must mirror |
| `search.py` | the random search of `docs/search_protocol.md`, a separate experiment with its own rules |
| `figures.py`, `cite_numbers.py` | the report's figure, and the table of every published number with its source |

`reproduce.sh` runs them in order and writes `results/`.

## The rest

The other scripts are the lab notebook in code: each one measured something
recorded in `docs/model.md` or `docs/status.md`, most of it a dead end. They are
kept because a negative result is only checkable if the code that produced it
is. `snapshot_corpus.py` and `snapshot_extend.py` build the frozen snapshots
from the project's database, which is not public.
