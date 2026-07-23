# Vertex Boxing — model

A port of the vertexmma bout-outcome model to boxing. The spine is identical;
the sport-specific parts differ. Read `docs/model.md` at the repo root for the
full "what ports / what dies / what's new" rationale.

## What ports unchanged
- Point-in-time, **leak-free** chronological replay (snapshot each fighter's
  aggregates strictly BEFORE the bout).
- Elo (K=32) + Glicko-2 conservative (rating − 2·RD); optionally WHR.
- Opponent-quality / strength-of-schedule aggregates — the **core** boxing
  signal (padded/protected records).
- LightGBM + CatBoost + LogReg + blender; A/B symmetrization; temporal split;
  rolling backtest.
- Closing line is **eval-only**, never a feature.

## What changes for boxing
- **3-outcome** target (win / draw / loss), multiclass. Evaluate with RPS +
  multiclass log-loss, and CLV on the competitive subset.
- **Drop** the per-round attack/defense skill ratings (no free punch data).
- Method taxonomy KO/TKO/UD/SD/MD/PTS/RTD/DQ (no submissions).
- Variable scheduled rounds (4/6/8/10/12) as a class proxy; re-fit age curve.
- New features: padded-record/SoS detector, opponent-adjusted KO power vs chin,
  hometown/venue decision bias × P(decision), activity/ring-rust, title level,
  amateur/Olympic pedigree, southpaw matchup, explicit draw sub-model.

## The kill-test (run FIRST)
`scripts/run_killtest.py` — replay leak-free, join historical closing odds,
and check for **positive CLV on competitive fights (implied 30–70 %)**. If
there's no edge there, the thesis fails regardless of headline accuracy. Run
this on the open-source bootstrap before building the full BoxRec pipeline.

## Layout (to be filled in the model step)
```
src/
  config.py           # version, split, params, competitive band   [done]
  db.py               # psycopg connection
  export.py           # raw -> leak-free per-bout feature rows (3-outcome)
  opponent_ratings.py # Elo + Glicko-2 + SoS aggregates (no punch stats)
  features.py         # row -> A/B diff matrix
  ensemble.py         # LGBM + CatBoost + LogReg + blender (multiclass)
  train.py / predict.py
scripts/
  run_train.py  run_predict.py  run_killtest.py
```
