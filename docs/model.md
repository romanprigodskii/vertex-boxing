# Model — the boxing port

Adversarially-verified plan (2026-07). The vertexmma model is honest and
mature; this documents exactly what carries over and what must change. The plan
below is kept as written; what follows immediately is what actually happened
when it was run.

## Where it stands (measured 2026-08-01)

One command reproduces it: `scripts/simulation/scripts/market_eval.py --blend`.
Three instruments, because the quoted set alone cannot resolve 0.005 and the
corpus alone is four fifths club boxing the market never prices.

| instrument | n | log-loss |
|---|---|---|
| corpus holdout | 103,611 | 0.3432 |
| premium holdout (sched ≥8, both ≥8 bouts) | 14,566 | 0.2967 |
| quoted, against the close | 1,838 | 0.4065 vs the market's 0.3658 |

The model loses to the closing line by 0.0398 and a blend of the two beats the
close by +0.0032 [+0.0005, +0.0060] under **all four** de-vig methods. Taking
the model's picks at the OPEN and marking them to the close is worth +0.0117 of
probability [+0.0090, +0.0145] over 1,475 bets — measure CLV in probability, not
in price, because this feed's open carries 5.8% margin and its close 7.9%, so a
random side already "beats the close" by 3.5% in price and an underdog-leaning
one by 5.3%.

**Two errors made the 31 July numbers look twice as good as they were.** The
profile crawl had targeted the fighters the odds feed quotes; those men stand in
the first corner and the first corner wins 87% of the time, so whether we
happened to know a man's birthday predicted the winner at 0.86 against 0.14 —
through nothing but a NaN, on 53% of the test set. And the de-vig was treated as
a matter of taste when the train slice settles it: the proportional close needs a
logit slope of 1.353 to be calibrated and the power close 1.078. Together they
were worth 0.034 of the reported gap. `scripts/leak_check.py` makes the first
test permanent; `--devig auto` makes the second automatic.

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
from even it is 0.4510 against 0.4486 — and the whole gap sits at the two ends
with the errors pointing opposite ways. On bouts the market calls 30-70% the
model scores 0.768 against a coin flip's 0.693; on 95% favourites it will not go
past 0.90. Inside the competitive band its calibration slope is 0.240 while on
the whole test set it is 0.939: calibrated on average, and collapsing exactly
where the price disagrees with it. Nothing in our own features identifies those
bouts — only the price does, and the price is eval-only.

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
