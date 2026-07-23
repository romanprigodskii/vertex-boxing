# Data sources

From a 12-agent research sweep (adversarially verified, 2026-07). Interactive
version: the [build dossier artifact]((a private working note)).

## Bootstrap — clean, legal, no anti-bot (pull first)
| Source | What | Access |
| --- | --- | --- |
| **Kaggle** `mexwell/boxing-matches`, `iyadelwy/…predict-winner` | ~50–300k bouts, A/B reach + pre-fight aggregates + judge scores | `kaggle datasets download`. **Prototype only** — stale (2019–23), BoxRec-provenance. |
| **Wikidata SPARQL** (CC0) | ~19k boxer entities, ~10k with **BoxRec-ID crosswalk (P1967)**, DOB/height/photo | `query.wikidata.org/sparql`. Identity spine + BoxRec-ID seeds. |
| **DBpedia SPARQL** (CC BY-SA) | ~15k W-L-KO career summaries | `dbpedia.org/sparql`. Join to Wikidata via `owl:sameAs`. |
| **Wikipedia record tables** (CC BY-SA) | The only **open fight-by-fight** source: opponent/date/result/method/round/venue for top ~1–3k | `api.php?action=parse`. |
| **boxing-data.com** (RapidAPI) | Clean JSON records/physicals/events/`/events/schedule`/rankings/titles | Free 100 req/mo → $29+. Legal cross-check + forward slate. |
| **Wikimedia Commons** | Portraits via P18, PD/CC0/CC-BY | `commons.wikimedia.org/w/api.php`. |

## Odds — the market yardstick
| Source | What | Notes |
| --- | --- | --- |
| **The Odds API** | Live h2h+totals (us/uk/eu) + historical since 2023-05-30 | Free 500 cr/mo. Clean backtest core. |
| **ProBoxingOdds.com** | Open+close moneylines since 2016, undercards + props | Light anti-bot, fits httpx+BS4. The workhorse. |
| Betfair BSP / Pinnacle-via-SportsGameOdds | The **sharp** price for true CLV | Paid; buy once kill-test looks promising. |
| OddsPortal | Pre-2023 depth | Cloudflare + JS → Playwright, later. |

Coverage caveat: the softest regional cards are largely **unpriced** — the
odds-covered subset skews liquid/sharp. Validate the regional-edge thesis with
forward paper-trading, not only historical backtest.

## The long tail — not used yet
**BoxRec** (~1.3M bouts, the regional/journeyman depth the thesis targets) is
Cloudflare-403-walled + login-gated, and its ToS forbids automated extraction
**and** redistribution of a BoxRec-derived database. All OSS scrapers are dead.
A slow, authenticated, account-risking backfill is a **separate later track**,
pursued only if the open-source kill-test greenlights the project — and even
then, internal model features only, never republished data.

**CompuBox** (per-round punch stats) is proprietary, televised-only, auth-gated
— design the model to not need it.
