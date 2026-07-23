# Model — the boxing port

Adversarially-verified plan (2026-07). The vertexmma model is honest and
mature; this documents exactly what carries over and what must change.

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
