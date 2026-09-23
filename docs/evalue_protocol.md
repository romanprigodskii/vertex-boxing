# Pre-registration: an e-value audit of the boxing model against Bet365's close

Roman Prigodskii · written 2026-09-23, committed before the audit it describes
was run · machine-readable part: [`ev_registry.py`](../scripts/simulation/scripts/ev_registry.py),
scoring: [`ev_audit.py`](../scripts/simulation/scripts/ev_audit.py)

The commit that adds this file, the registry and the audit script is the
timestamp. Anything changed after it is listed in the paper as a deviation, with
its reason and its diff.

## 1. The question

The final report ([REPORT.md](REPORT.md)) scored a frozen model against one
price feed, ProBoxingOdds, and found two things: on its own the model loses to
the closing line (−0.0211 nats), and blended into the price it improves it
(+0.0043 at λ 0.17). This audit asks the same model, unchanged, three questions
against a second bookmaker's line, and in a currency that pays for how many
questions are asked:

1. Does the blend beat Bet365's close, and on bouts no price in this project has
   ever been scored on?
2. Can the same knowledge be turned into money at the prices Bet365 actually
   posted, at the open or at the close?
3. Is there a slice of boxing, or a side of a bout named before the bell, where
   the closing price is wrong?

## 2. Data, and what had been seen before this commit

**Model.** The published model of REPORT 4.1, trained on the corpus up to
2023-06-10. Its predictions on every non-drawn holdout bout after that date are
saved in `imports/staging/preds/final-close.npz` (`p_corp`, `key_corp`); on the
3,288 priced bouts they equal the scoreboard's to the last digit. Nothing is
retrained.

**Prices.** BetsAPI's archive of Bet365 boxing lines, fetched 2026-09-23
(`backfill_betsapi.py`). The close is `kickoff`, Bet365's last price before the
bout went in-play; another book's last price counts only if stamped before that
moment. The reference book of each bout is Bet365 wherever it priced the bout.
Joined to the corpus by `market_eval.join_odds`, the same join the report uses.

**What was seen before this commit**, all of it on 2026-09-23:

- Raw prices of about ten bouts while the feed's layout was being read: four on
  2023-08-12 and 2025-09-20, the rest in September 2026. No outcome was scored.
- **The first seven months of 2026** (632 bouts, 2026-01-16 → 2026-07-24):
  the model against Bet365's close in aggregate, by scheduled distance, on the
  bouts shared with ProBoxingOdds and on the rest, and the published λ = 0.17
  blend applied to Bet365. These bouts are **outside the confirmatory window**
  (section 5).
- The null simulation of section 6 on the prices of 2025-11-21 → 2025-12-31,
  which draws outcomes from the price and reads no real one.

**What had been seen long before, and cannot be un-seen.** About 70% of the
confirmatory bouts are also priced by ProBoxingOdds, and the report analysed
those bouts at length: the gap and the blend, closing-line value by distance,
belt and record depth, and λ by slice. On the bouts both feeds share, the two
closes agree on the favourite 98.5% of the time (median gap between them 0.013
in probability, measured on the 2026 slice). **So on shared bouts this audit is
a new instrument on old outcomes, not new evidence.** The slices were chosen
knowing the report. The only bouts whose outcome has never been scored against
any price here are the ones ProBoxingOdds does not price, and they get their own
primary hypothesis (P2).

## 3. The null and the bet

For each bout, q is the probability of corner A winning in Bet365's closing
price, the margin removed by the power method. That is fixed here, because it is
the method whose de-vigged close is calibrated on the published run's training
slice (slope 1.02). **H0: q is the probability that A wins, given everything known
before the bell.** Draws are void, because the market is two-way.

Against an alternative probability p′, full Kelly at fair odds multiplies wealth
by p′(y)/q(y). Its expectation under H0 is 1, so the product over a set of bouts
is a test martingale, and its value at any stopping time is an e-value (Ville's
inequality). Its logarithm is the log-loss improvement of p′ over q summed over
the bouts. That is the scoreboard's own quantity, turned into a bet.

The alternative is never one choice made after the fact. It is a uniform mixture
over a grid fixed in the registry, and a mixture of test martingales is a test
martingale:

- **M hypotheses (the model knows something here):** p′ = σ(λ·logit p +
  (1−λ)·logit q), λ ∈ {0.05, 0.10, …, 0.50}.
- **B hypotheses (the price is biased against a named side, no model):** that
  side's probability moved by δ logits, δ ∈ {0.05, …, 0.50}, in the direction
  the stated mechanism predicts. Where two mechanisms point opposite ways, both
  directions are in the mixture, which pays for them.
- **Control C1:** the price recalibrated, σ(β·logit q), β ∈ {0.80, …, 1.20}
  without 1. It carries no model. If it comes out as large as P1, then P1 has
  measured the de-vig, not the model.

**Two currencies for every hypothesis.**

- *Fair* is the process above.
- *Real* stakes Kelly on the same alternatives at the decimals Bet365 actually
  posted, which bets only where p′ times the price exceeds 1. At posted prices
  every bet has expectation at most 1 for any true probability inside the band
  the two prices leave, so *real* is an e-value against that composite null. It
  has nothing de-vigged in it, and it is the null a bettor faces.

Stakes never depend on earlier outcomes: p, q and the grids are all fixed before
each bell. So the terminal e-value does not depend on the order the bouts are
multiplied in.

## 4. The hypotheses

**Primary: four hypotheses, a union bound, rejected at e ≥ 4/α = 80.**

| | set | alternative | currency decided on |
|---|---|---|---|
| P1 | every confirmatory bout | M | fair |
| P2 | confirmatory bouts ProBoxingOdds does not price on any date | M | fair |
| P3 | confirmatory bouts with an opening price | M, blending the model with the **opening** price | real, at the opening decimals |
| P4 | every confirmatory bout | M | real, at the closing decimals |

C1 is reported beside P1 and decides nothing.

**The family: K = 61, e-BH at α = 0.05** (Wang and Ramdas, 2022). e-BH controls
the false discovery rate under arbitrary dependence, and these slices are
overlapping subsets of one pool, scored by one model against one book. The
registry holds each hypothesis's filter, side and stated mechanism:

- 45 M slices:
  - the level axis the thesis was about: scheduled distance, belt, main event,
    and the report's own upper and lower tiers;
  - record depth and experience mismatch;
  - form: a win streak of 10 or more, both fighters unbeaten, coming off a loss
    or a stoppage, layoffs;
  - geography: same country, home and away, venue country;
  - weight;
  - the price itself: favourite strength, line movement, how far model and
    price disagree;
  - rematches;
  - three calendar periods.
- 16 B biases:
  - favourite-longshot, overall and in mismatches;
  - a fighter on a 10+ win streak, an unbeaten 10+-bout record, the home
    fighter, the British fighter at home;
  - a fighter coming off a stoppage loss, a long layoff, lopsided experience;
  - the side the line moved towards, the world-title favourite, the heavyweight
    underdog, the streaking underdog, the harder puncher, the cross-border
    favourite, the club favourite.

e-BH is applied twice, as two separate claims: to the fair e-values (the price is
wrong here) and to the real e-values (a bettor could have made money here).

**The count, as a global test.** Added 2026-09-23, still before any real
outcome was scored. It asks how many of the 61 reach e ≥ 20, and it reads that
number against the distribution of the same count in the null simulation of
section 6, never against "5% of K". The slices overlap and move together, so
under the null the count is spread wider than a binomial. A count above the
simulated 95th percentile says there is something in the family somewhere. It
does not say which slice, which is what e-BH is for.

## 5. Windows

- **Confirmatory: 2023-06-11 → 2025-12-31.** Every decision in section 4 is taken
  here, at its end.
- **Continuation: 2026-01-01 → 2026-07-24,** which was seen in aggregate
  (section 2). The processes are carried through it and reported, labelled as
  such. No decision uses it.

## 6. Validation before use

Before the real run, the audit is run with outcomes drawn from q itself, on the
confirmatory prices, with at least 1,000 replications (`ev_audit.py --null`). It
passes if:

- the mean e-value of every hypothesis is within Monte Carlo error of 1 or below;
- no single hypothesis crosses 1/α more often than α;
- e-BH rejects anything in at most α of the runs.

If it fails, the audit is not run until the instrument is fixed, and the fix is
a deviation.

## 7. What will be reported, whatever it says

- Every e-value, fair and real, in both windows, including the ones that
  decide nothing.
- The ladder for every primary: the fixed-λ published blend (λ 0.17), the
  mixture, fair, real.
- Sensitivity analyses, none of which changes a decision:
  - the proportional de-vig;
  - ProBoxingOdds' close instead of Bet365's, on the shared bouts;
  - the growth rate per bout, and the number of bouts it would take to reach 1/α.

## 8. Known limitations, stated before the result

- The shared-bout overlap of section 2. It is the largest limitation, and P2 is
  the answer to it.
- The model is frozen at 2023-06-10, so it is two and a half years stale by the
  end of the window. A deployment would retrain yearly (REPORT 4.1: −0.0198
  against −0.0211).
- Win streaks and last results come from the corpus alone, which can miss a bout.
  On the test window they agree with the event page's form strip 98% of the time
  where the strip is shorter than six.
- About a third of closes on the part of the archive fetched so far are a
  book's last price before the bell (`end`, or `start` if it never moved)
  rather than a `kickoff`, because Bet365 had no in-play market for the bout.
  These are pre-bell by the rule of section 2, but not pre-bell to the minute.
  The final share is reported.
- One sport, one bookmaker, one model.
