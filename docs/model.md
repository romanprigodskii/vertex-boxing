# Model — the boxing port

Adversarially-verified plan (2026-07). The vertexmma model is honest and
mature; this documents exactly what carries over and what must change. The plan
below is kept as written; what follows immediately is what actually happened
when it was run.

## Where it stands (measured 2026-08-01, second pass)

One command reproduces it: `market_eval.py --tta --blend`. The defaults ARE the
best known model. Three instruments, because the quoted set alone cannot resolve
0.005 and the corpus alone is four fifths club boxing the market never prices.

| instrument | n | log-loss | was | paired gain |
|---|---|---|---|---|
| corpus holdout | 89,087 | **0.3346** | 0.3390 | +0.0044 [+0.0038,+0.0051] |
| premium holdout (sched ≥8, both ≥8 bouts) | 12,536 | **0.2859** | 0.2901 | +0.0042 [+0.0025,+0.0060] |
| quoted, against the close | 3,288 | **0.3729** vs the market's 0.3469 | 0.3799 | +0.0070 [+0.0032,+0.0107] |

The gap to the closing line is **−0.0260**, down from −0.0329: a fifth of it
closed. The blend now beats the close by **+0.0041 [+0.0020, +0.0062]** (λ 0.18,
up from 0.15 — the price has less to say about the model than it used to), and
clean CLV rose from +0.0106 to **+0.0114 of probability [+0.0092, +0.0135] over
1,979 bets** at a 2% edge.

### What the second pass changed, in order of size
1. **87 new features** (`everyx`, 200 columns), +0.0025 on the confirmation half
   [+0.0017,+0.0033]. Not one of the six groups is worth anything alone; the
   block is. What carries it is `cmp` (+0.0016) — do these two ratings even
   come from the same graph — and `thin` (+0.0006).
2. **Test-time symmetrisation** (`--tta`), a further +0.0017 on the confirmation
   half and +0.0030 on the quoted set. A boxing match has no corner A, but the
   model's answer depended on which name was typed first; asking it both ways
   and averaging the two logits cancels the half of that which is noise. One
   extra forward pass, no retraining.
3. **Training on both orientations** (`--mirror`), +0.0016 more at 2.5× the
   training time. Measured on one seed.
4. **Retraining as time passes** (walk-forward, 12 months), +0.0019 on the
   confirmation half and +0.0000 on the selection half — which is the signature
   of staleness and not of a better model, since the selection half is the year
   right after the cutoff and the confirmation half is two years later. One
   seed. Deployment would do this anyway.

### The three defects the mirror found
Building the mirrored matrix by a second replay rather than by negating columns
made a test possible that had never been run: every column must be a difference
that negates, a quantity invariant to the swap, a probability that becomes 1−p,
or half of a pair that trades with its twin. `mirror_check.py` runs it and
found three columns that failed — `city_home_bias`, `jud_fav` and the
`d_elo_jud` built on it. All three were tie-breaks written `>=`, so when two
fighters had equal ratings (or neither was local) the update fell through to
"corner A" and a quantity that is a fact about a city or a judge came out
different depending on which way round the bout had been written down. Fixed by
refusing to update when there is no favourite and no local man.

### The board has three prices and they answer different questions

| reading | margin | market | model | gap | λ | blend over market |
|---|---|---|---|---|---|---|
| worst price on the board | 6.2% | **0.3469** | 0.3729 | −0.0260 | 0.18 | +0.0041 [+0.0020,+0.0062] |
| best price on the board | 3.0% | 0.3527 | 0.3774 | −0.0247 | 0.19 | +0.0044 [+0.0019,+0.0068] |
| opening line | 5.6% | 0.3591 | 0.3729 | −0.0138 | 0.37 | +0.0096 [+0.0053,+0.0139] |

(was −0.0329 / −0.0306 / −0.0208 at λ 0.15 / 0.16 / 0.33 before the second pass;
λ rising is the point — the price now accounts for less of what the model knows.)

**The best price is the worst forecast.** Best-of-market is not the market's
opinion; it is the upper envelope over ten books, so taking the maximum on both
sides picks out the two books that most disagree with consensus. Bet at the best
price, score against a balanced one. The blend beat the market under all four
de-vig methods on the best price before the second pass (power +0.0040, Shin
+0.0046, additive +0.0045, proportional +0.0062) and the second pass moved the
power reading, the one `--devig auto` picks, to +0.0044. The other three have
not been re-measured on the new model; they were never the binding case.

### The two results that are not circular
The model never sees a price, so its closing-line value is clean: taking its
picks at the OPEN and marking to the close is worth **+0.0114 of probability
[+0.0092, +0.0135] over 1,979 bets** at a 2% edge (was +0.0106 before the second
pass; +0.0152 at a 5% edge over 1,323). Flat-staked ROI into the real closing
number also improved at every threshold — −6.30% → −3.25% at a 2% edge,
+5.10% → +6.41% on competitive bouts — though none of those intervals excludes
zero on this sample.

Measure CLV in PROBABILITY. The price-space null — bet a random side at the open,
mark to the close — is +3.30%, which is the two margins and nothing else.

Measure CLV in PROBABILITY. The price-space null — bet a random side at the open,
mark to the close — is +3.5% reading the close as the worst price and −4.9%
reading it as the best. That is the two margins and nothing else. And never
compute CLV for the blend against any closing price: it takes one as its input
and beats it by construction. That guard has failed twice here, first by being
absent and then by testing the label `== "close"` and letting the best-price run
through with a spurious +18% ROI.

### Three defects in the yardstick, each of which flattered us
1. The profile crawl targeted the fighters the odds feed quotes; those men stand
   in the first corner and the first corner wins 87% of the time, so whether we
   happened to know a man's birthday predicted the winner at 0.86 against 0.14 —
   through nothing but a NaN, on 53% of the test set. `scripts/leak_check.py`
   makes the test permanent.
2. The de-vig was treated as a matter of taste when the train slice settles it:
   proportional needs a logit recalibration slope of 1.297 and power 1.021.
   `--devig auto` picks the calibrated one.
3. "Closing range" is three columns — the worst and the best price across ten
   books — and the parser read only the worst. The "7.9% bookmaker margin" this
   project quoted was never a margin; it was the width of the books'
   disagreement. And the crawl's target list was built against the abandoned
   Wikipedia layer, so it asked for 1,277 of 9,691 pages: the source was not 87%
   uncrawled, it was 87% never requested. Re-crawled to 23,666 matchups, which
   took the quoted test set from 1,838 bouts to 3,288.

**Numbers from before the re-crawl are on a different population.** The cutoff is
the 60th percentile of quoted dates, so tripling the odds file moved it from
2022-11-11 to 2023-06-10 and the corpus holdout from 103,611 bouts to 89,087.
0.3432 → 0.3390 is not a measurement of anything.

### What each group is worth (premium holdout, leave-one-out, paired bootstrap)
Glicko-2 +0.0042 · referee +0.0016 · judges' scorecards in the ratings +0.0018 ·
durability and mileage +0.0016 · level of bout +0.0015 · head-to-head and common
opponents +0.0014 · "both profiles known" +0.0018.

The six groups added on 2026-08-01, leave-one-out on the corpus holdout at one
seed (so ±0.001, and only `cmp` is clear of it):

| group | what it is | dropping it costs |
|---|---|---|
| `cmp` | same passport, venue-histogram cosine, days since a stoppage loss, journeyman index, rating z-score against the population active that month | **+0.0016** |
| `lvlr`+`lvlq` | the LEVEL of every rating and rate, not only the difference — min and max over the two corners | +0.0010 |
| `thin` | one-sided-tolerant records, and what the card says when one man has none | +0.0006 |
| `res` | performance against what Elo expected, and the fall from a career peak | +0.0006 |
| `ctx` | running upset rate by country, promoter, division, distance | +0.0005 |
| `unc` | Glickman's expected score, gaps divided by their own standard error | +0.0000 |

Read that table as a whole, not row by row: none of the six is worth anything on
its own (each scored +0.0000 to −0.0002 when added to `everyc` alone), and
together they are worth +0.0025. They are six ways of saying the same thing —
how much should a rating gap be trusted HERE — and the model needed enough of
them at once to tell the cases apart.

### The three instruments, and why a one-seed screen cannot see 0.001
The corpus holdout is split in half by date: the EARLY half selects, the LATE
half reports, and nothing is ever chosen on the late half (`lab.py`). And seed
noise is not bout noise — a paired bootstrap over bouts does not see it at all.
The same configuration on three seed sets scores 0.3375 / 0.3377 / 0.3379 on the
corpus and 0.3326 / 0.3332 / 0.3336 on the confirmation half, so **a one-seed
screen resolves about 0.001 and no better**. Everything above 0.002 was
re-measured on five seeds before it was believed; everything at 0.001 is
reported as one seed and labelled as such.

### Measured dead ends — do not re-run these
Isotonic or Platt calibration of any population (isotonic on 29k club bouts cost
0.003 on the quoted set) · calibration whose slope varies with the matchup's own
uncertainty, +0.0000 [-0.0001, +0.0002] · a blend whose λ fades with the
model-market disagreement · reshaping the target in any of four ways
(branch decomposition, margin regression, soft labels, three-class), all
+0.0000 · latent/non-transitive style ratings — the corpus has 48,151 triangles
and a median fighter with two bouts · **hyper-parameters**: a 500-trial TPE
search tuned on a window ending three years before the reporting holdout gained
0.0025 on its own window and 0.0000 on the holdout, all of it selection tax
(`scripts/simulation/scripts/tune.py` holds the protocol) · weigh-in weights ·
judges' home bias, even with real country flags · belts and card position, which
level of bout had already saturated · CatBoost/LogReg ensembling · monotone
constraints · training on the premium population only.

### The diagnosis, rewritten from the second pass
The old diagnosis — "the model is over-confident where the price says pick'em" —
was reading a per-bout table without weighting it by how many bouts are in each
band. Weighted, the gap is nearly FLAT across the price scale (+0.0066, +0.0073,
+0.0043, +0.0070, +0.0072 of the total, from 95% favourites down to coin flips).
And the model is not miscalibrated: its slope is 0.989 on the corpus holdout,
0.998 on the premium one, 0.943 on the quoted set. Shrinking every logit by a
factor chosen *on the test set itself* — an oracle no honest model gets — buys
0.0005. **There is no calibration fix, because there is nothing to calibrate.**

What there is instead, from `where.py`:

- **77% of the gap is on bouts where the MARKET is bolder than we are**, not the
  other way round (0.3996 against 0.3449 there, against 0.3628/0.3487 where we
  are bolder). The failure is missing information, not misplaced confidence.
- **Where our own evidence runs out, it runs out completely.** On the 205 bouts
  (6.2%) where the less experienced man has fewer than three recorded bouts, the
  market scores 0.1379 and we score 0.2491 — **19.7% of the whole gap**. A
  debutant priced at 97% is an amateur international and the corpus has never
  heard of him.
- **Where we say "coin flip" the market usually is not guessing**: on the 373
  bouts where our |logit| is under 0.5, the market scores 0.5986 to our 0.6889 —
  31% of the gap.

### What would move the needle next
1. **Amateur pedigree.** It is the largest addressable block on the board: a
   fifth of the deficit sits on debutants and near-debutants, and BoxRec carries
   an amateur tab while Wikidata carries Olympic medals. This is a crawl, not a
   feature.
2. **Walk-forward in production.** +0.0019 on the second half of the holdout for
   nothing but a cron entry, and it grows with the age of the model.
3. **The yardstick.** proboxingodds.com's front page is a LIVE ten-book board
   including Polymarket and Kalshi — near-zero-margin prediction markets, a far
   sharper benchmark than any historical close — and it exists only going
   forward: nothing about it can be recovered after the fact. Forward capture is
   at zero of five parts.
4. Not hyper-parameters. Re-checked on the real protocol: 63→31→127→255 leaves,
   lr 0.03→0.015, min_data 30→300, λ₂ 5→30, feature fraction 0.9→0.6 — the best
   of them is worth +0.0006 on the confirmation half and none has an interval
   clear of zero. And not the quoted-population reweighting (−0.0019), nor
   folding draws into the target (+0.0000), nor Glickman's expected score as a
   boosting offset (+0.0001).

### The four scripts that hold the method
| script | what it is for |
|---|---|
| `market_eval.py` | the scoreboard. One command, three instruments, the blend |
| `lab.py` | the bench. One data load, many variants, a paired verdict each; the holdout split into a half that selects and a half that reports |
| `mirror_check.py` | every column of the mirrored matrix must negate, stay put, become 1−p, or trade with its twin. Found three real defects the first time it ran |
| `leak_check.py` | no feature's missingness may name a corner |
| `where.py` | where the market's advantage sits on axes we can see before the bell |

## The honest verdict
The thesis (softer boxing market) is **directionally right but not a free
lunch**. Soft regional lines exist, but softness tracks *low liquidity*: the
softest fights carry $100–500 limits, 4.5–8 % regional vig, and winners get
limited fast (Kaunitz et al. 2017: +6.2 % over 672 bets, then account-limited).
Realistic upside is low-single-digit ROI, high variance, capacity-constrained.

## Metric: CLV, not accuracy
On regional cards the favorite wins ~90–95 % (deliberate padded matchmaking),
so "always pick the favorite" scores ~90 % and accuracy is meaningless. Judge
the model only on:
1. **Closing-line value on the competitive subset** (market-implied 30–70 %).
2. Calibration / multiclass log-loss / RPS.

Walsh & Joshi (2024): selecting models by calibration vs accuracy = +34.7 % vs
−35.2 % ROI. Ratings-only accuracy ceiling is ~60 % (BoxMind 2026); ~70 %
needs footage-derived punch data we won't have.

## Ports from vertexmma unchanged
Point-in-time leak-free replay (matters more here — padded records make leakage
more misleading) · Elo (K=32) + Glicko-2 conservative (rating − 2·RD) ·
opponent-quality / strength-of-schedule Elo aggregates (the **core** signal) ·
record/finish/layoff/form/streak/age/reach/stance features · LGBM + CatBoost +
LogReg + blender · isotonic calibration · A/B symmetrization · temporal split +
rolling backtest · **closing line eval-only, never a feature**.

## Must change
- **Drop** the online opponent-adjusted attack/defense skill ratings + all
  takedown/submission/control features — no free per-round boxing punch data
  (CompuBox is proprietary, televised-only, ToS-restricted).
- Target → **3-outcome win/draw/loss** (Davidson tie term; Davidson-Beaver home
  term). Draws are frequent enough to bias a binary model.
- Methods → KO/TKO/UD/SD/MD/PTS/RTD/DQ (no submissions).
- 17+ divisions + catchweights + weight-jumps; variable scheduled rounds
  (4/6/8/10/12) as a class proxy, not a constant.
- Re-fit the age curve (boxing peaks ~28–35, weight-dependent; add
  cumulative-rounds "mileage"). Prefer Glicko-2/WHR over plain Elo for the
  sporadic, strategic schedules; consider adding WHR (BoxRec's own method).

## New for boxing (by signal-per-obtainability)
1. **Padded-record / strength-of-schedule detector** beyond avg-opponent-Elo —
   journeyman-opponent density, recursive QOO/QOOP graph SoS, "beaten anyone
   with a winning record at level X" flags.
2. **KO power vs chin**, opponent-adjusted — finish-for / finish-against,
   never-been-stopped flag, rounds-per-bout (partially recovers lost skill
   signal).
3. **Hometown/venue decision bias** × P(bout reaches a decision) — home-win
   prob 0.57 KO / 0.66 TKO / 0.74 points (PubMed 16089185).
4. Activity / ring-rust + cumulative-rounds mileage.
5. Rounds-scheduled + title/sanctioning level as class proxies.
6. Amateur/Olympic pedigree (Sherdog-analog for debutants) · southpaw-vs-
   orthodox matchup interaction · explicit draw sub-model · weight-jump /
   catchweight · promoter identity.

## The kill-test (do this first)
Assemble 12–24 months of results + historical closing odds, replay leak-free,
and check for positive CLV on the competitive subset at realizable limits. If
it can't beat the close there, the port fails regardless of accuracy — and
you've spent a night, not a month. `scripts/simulation/scripts/run_killtest.py`.
