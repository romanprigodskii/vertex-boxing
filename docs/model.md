# Model — the boxing port

Adversarially-verified plan (2026-07). The vertexmma model is honest and
mature; this documents exactly what carries over and what must change. The plan
below is kept as written; what follows immediately is what actually happened
when it was run.

## Where it stands (measured 2026-08-01)

One command reproduces it: `scripts/simulation/scripts/market_eval.py --blend`.
The defaults ARE the best known model. Three instruments, because the quoted set
alone cannot resolve 0.005 and the corpus alone is four fifths club boxing the
market never prices.

| instrument | n | log-loss |
|---|---|---|
| corpus holdout | 89,087 | 0.3390 |
| premium holdout (sched ≥8, both ≥8 bouts) | 12,536 | 0.2901 |
| quoted, against the close | 3,288 | 0.3799 vs the market's 0.3469 |

### The board has three prices and they answer different questions

| reading | margin | market | model | gap | blend over market |
|---|---|---|---|---|---|
| worst price on the board | 6.2% | **0.3469** | 0.3799 | −0.0329 | +0.0036 [+0.0017,+0.0054] |
| best price on the board | 3.0% | 0.3527 | 0.3833 | −0.0306 | +0.0040 [+0.0018,+0.0061] |
| opening line | 5.6% | 0.3591 | 0.3799 | −0.0208 | +0.0088 [+0.0045,+0.0130] |

**The best price is the worst forecast.** Best-of-market is not the market's
opinion; it is the upper envelope over ten books, so taking the maximum on both
sides picks out the two books that most disagree with consensus. Bet at the best
price, score against a balanced one. The blend beats the market under all four
de-vig methods on the best price (power +0.0040, Shin +0.0046, additive +0.0045,
proportional +0.0062).

### The two results that are not circular
The model never sees a price, so its closing-line value is clean: taking its
picks at the OPEN and marking to the close is worth **+0.0111 of probability
[+0.0091, +0.0130] over 2,234 bets**, replicating +0.0117 measured on a smaller
set. And the first ROI interval to exclude zero on a real sample: competitive
bouts (market 30–70%), a 2% edge, settled at the best price — **+10.59%
[+0.74%, +20.54%] over 433 bets.**

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

### The diagnosis that is still open
The model equals the market in the middle of the scale — on bouts priced 10-20%
from even it is 0.4682 against 0.4590 — and the whole gap sits at the two ends
with the errors pointing opposite ways. On bouts the market calls 30-70% the
model scores 0.713 against a coin flip's 0.693; on 95% favourites it will not go
far enough. Inside the competitive band its calibration slope is 0.240 while on
the whole test set it is 0.939: calibrated on average, and collapsing exactly
where the price disagrees with it. Nothing in our own features identifies those
bouts — only the price does, and the price is eval-only.

### What would move the needle next
Not features: the last three groups added were worth +0.0008 on the premium
holdout between them, and hyper-parameters are exhausted. The binding constraint
is the yardstick. proboxingodds.com's front page is a LIVE ten-book board that
includes Polymarket and Kalshi — near-zero-margin prediction markets, a far
sharper benchmark than any historical close — and it exists only going forward:
nothing about it can be recovered after the fact. Forward capture is at zero of
five parts, and every day it stays there is a day of the only benchmark that
would settle the thesis.

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
