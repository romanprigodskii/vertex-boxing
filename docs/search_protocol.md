# A large random search, and the rules it runs under

Written on 2026-09-21, **before the first fit**. `scripts/search.py` implements
this file and nothing else, and records this file's SHA-256 in every result it
writes, so the protocol a result was produced under can always be checked.

## The question

The model loses to the closing line by 0.021 nats on 3,288 priced bouts. Every
hyperparameter and weighting change measured so far has been worth at most a few
thousandths, and a 500-trial TPE search in August gained 0.0025 on its own window
and 0.0000 on the holdout. The proposal (Roman's): spend more compute, try far
more configurations, and re-check the ones that stand out.

That is sound if and only if the re-check is on data the search never saw. So
the question is narrow: **does any configuration found by a wide random search
beat the final model on bouts the search never saw, and does it change the
market comparison?**

## The windows, fixed now

| role | bouts | what it is used for |
|---|---|---|
| training for the search | every decisive bout dated on or before **2021-06-10** | fitting each candidate (early stopping on its last tenth, as everywhere on the bench) |
| **window A, selection** | decisive bouts after 2021-06-10 up to **2023-06-10**, and their premium subset | the only outcomes the search sees |
| **window B, confirmation** | the holdout after 2023-06-10: 89,087 bouts, its premium subset, and its two halves | the re-check; never seen by the search |
| the priced test | the 3,288 priced bouts inside window B | looked at once, at the end, for the configurations carried forward; never used to choose |

Window A is the same window `rule_oos.py` tests the level rule on. Nothing in it
overlaps window B.

## The search space

Each candidate draws every setting independently, from a random generator with
seed 20260921:

| setting | values |
|---|---|
| `num_leaves` | 15, 31, 63, 127, 255 |
| `learning_rate` | log-uniform on [0.01, 0.1] |
| `min_data_in_leaf` | 20, 50, 100, 200, 400 |
| `feature_fraction` | uniform on [0.5, 1.0] |
| `bagging_fraction` | uniform on [0.5, 1.0] |
| `lambda_l2` | log-uniform on [0.1, 50] |
| `extra_trees` | on, off |
| `max_bin` | 63, 255 |
| `path_smooth` | 0, 1, 10, 50 |
| half-life of the training weights | 3, 4, 6, 8, 12 years, or none |
| training weights by population | none; the premium population (≥ 8 scheduled rounds, both men ≥ 8 bouts) ×1.5, ×2 or ×3; the top tier (12 rounds or a continental or world belt) ×2 or ×4 |

The feature set is fixed at `everyz`, the final model's 231 columns.

## The screen

- **N = 200 candidates** on this machine overnight. If the screen gives a reason
  to scale out (next section), the same protocol runs with a larger N elsewhere,
  and the seed and the space do not change.
- Each candidate is fitted once, one seed, with test-time averaging over both
  orientations and no mirror training. That is cheap enough to screen with, and
  it is a different experiment from the deployment stack. That is why the
  confirmation refits in the full stack.
- **Ranking metric: log-loss on window A's premium subset.** The premium
  population is the one the market prices. Corpus log-loss on window A is
  recorded as a secondary.
- **The seed band.** The final configuration is screened the same way under five
  different seeds. The spread of those five is the noise a one-seed screen cannot
  see through. A candidate whose lead over the final configuration is inside
  that band has shown nothing.

## The confirmation

The **top 5** candidates by window-A premium log-loss, and the final
configuration as the baseline, are refitted to 2023-06-10 in the full deployment
stack: mirror training, test-time averaging, three seeds. Each is scored on
window B (corpus, both halves, premium) and on the priced test against the
closing price. Each candidate is compared with the baseline by a paired bootstrap
on the same bouts.

## The decision rule, fixed now

A candidate **survives** if its improvement over the baseline on **window B's
premium subset** has a **99%** paired-bootstrap interval that lies above zero
(Bonferroni for five candidates at 5%), and its improvement on the whole of
window B's corpus is not negative.

- If none survives: the search found nothing. That is the expected result, and
  it is reported as a result.
- If one survives: its priced-test comparison with the market is reported
  whatever it shows, and it is **not** adopted as the new final model on the
  strength of one search. It would need its own confirmation on data collected
  after today.
- In every case the winner's shrinkage is reported: its lead on window A against
  its lead on window B.

## When a larger search is worth running

Only if the screen shows candidates whose lead on window A is well outside the
seed band **and** at least one of them survives confirmation. A larger N on its
own raises the best screened lead by selection alone. The confirmation is what
tells a real lead from that, and it does not get easier with more candidates.
