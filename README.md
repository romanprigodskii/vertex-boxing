# Vertex Boxing

AI-powered boxing fight-prediction platform — a port of [vertexmma](../vertexmma).
Predicts professional boxing bouts and grades the model against the closing
betting line, with the thesis that **soft regional/undercard markets** leave
room for a fundamentals model to find edge (where the razor-sharp UFC market
does not).

> **This is a falsifiable research bet, not a funded edge.** Success is measured
> as **closing-line value on competitive fights**, not accuracy — the ~90–95 %
> favorite base rate on regional cards makes accuracy meaningless. Run the
> [kill-test](scripts/simulation/README.md) before building the full pipeline.

## Stack
- **Next.js + TypeScript** web app (to be built) · **Drizzle ORM** + **Supabase** (Postgres)
- **Python** ingestion (`httpx` + `BeautifulSoup`) and modeling (LightGBM + CatBoost)

## Data — open sources only (for now)
BoxRec has the regional long tail but is Cloudflare-walled and its ToS forbids
extraction/redistribution, so the bootstrap uses clean, license-safe sources:
Wikidata (identity + BoxRec-ID crosswalk), DBpedia (record summaries),
Wikipedia record tables (fight-by-fight), boxing-data.com API, Kaggle dumps
(prototype only), plus The Odds API + ProBoxingOdds for the market line.
See [`docs/data-sources.md`](docs/data-sources.md).

## Model
Ports the leak-free point-in-time replay, Elo/Glicko-2, strength-of-schedule
aggregates, ensemble and calibration; drops the per-round punch-stat features
(no free CompuBox equivalent); goes 3-outcome (win/draw/loss); adds
boxing-specific signals (padded-record detection, hometown/venue decision bias,
title level, amateur pedigree). See [`docs/model.md`](docs/model.md).

## Layout
```
src/lib/db/schema/     # Drizzle schema (fighters, events/bouts, odds, rankings, predictions)
scripts/scraper/       # open-source data ingestion
scripts/odds_scraper/  # betting lines (the yardstick)
scripts/simulation/    # the model + the kill-test
docs/                  # model & data-source rationale
```

## Setup
```bash
pnpm install
cp .env.example .env.local   # DATABASE_URL + Supabase + ingestion keys
pnpm db:push                 # apply the schema
```
Then per-package Python venvs — see each `scripts/*/README.md`.
