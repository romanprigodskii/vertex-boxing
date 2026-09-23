# Pre-registration 2: money at the opening price, on a window nobody has scored

Roman Prigodskii · written 2026-09-23. This file, `ev_window.py` and
`market_eval.py` are hashed into `docs/evalue_protocol_2.manifest`. That
manifest is timestamped with OpenTimestamps and pushed to the public repository
**before** either model below is trained and before any outcome in either window
is read. The first pre-registration had only local commit times as evidence, and
anyone can set those. This one has two timestamps that no one can edit: a
Bitcoin block, and GitHub's record of the push.

## 1. Why this test

The first audit found one result that held up. Its protocol
(`evalue_protocol.md`) and its results (`results/evalue_audit.json`,
`results/evalue_followup.json`) are in the private working repository for now;
they will be published with this one's outcome. A model that knows only what was known when the line
opened, blended with Bet365's opening price and staked by Kelly at the posted
decimals, grew wealth by e ≈ 10⁸ over 2023-06 → 2025-12. It did the same at
ProBoxingOdds' opening price. Every check that could have killed it failed to.

What it could not rule out is that the window had been seen. The report had
already analysed those bouts, against another book's price. **This protocol
asks the same question, with the same instrument, on bouts where, as far as the
repository records, no one has ever scored the model against an opening price.**

## 2. The test

**Window A, the primary: 2016-06-11 → 2020-06-10.**

- **The model** is retrained on the corpus up to 2016-06-10 only, in the published
  configuration, without the features settled in fight week (weigh-in,
  referee, the card's judges, running order):

  ```
  VERTEX_ODDS=<repo>/imports/staging/proboxingodds_v3.parquet \
  python3 scripts/market_eval.py --tag l6 --tta --mirror --xt --seeds 3 \
      --cutoff 2016-06-10 --drop weigh+ref+judc+cardpos --label win-a-model
  ```
- **Prices:** ProBoxingOdds' opening line, `proboxingodds_v3.parquet`, the
  re-crawl with the fixed parser. Bet365 has no boxing prices before 2020-12.
- **Bouts:** every bout in the window that has an opening price. Bouts where
  either fighter boxed in the 60 days before are removed, because such a bout
  could fall between the opening price and the bell, and the model would know
  its result. Measured on Bet365, lines open a median 3 days before the bell,
  and 90% open no earlier than 37 days before. By dates alone, 79% of the
  window's 2,566 priced bouts pass.
- **The e-value:** q is the opening price for corner A, the margin removed by
  the power method. The alternative is p′ = σ(λ·logit p + (1−λ)·logit q),
  λ ∈ {0.05, 0.10, …, 0.50}, mixed uniformly. The stake is Kelly at the posted
  opening decimals. At posted prices every bet has expectation at most 1 for any
  true probability inside the band the two decimals leave, so this is an
  e-value against that composite null. Nothing is de-vigged in the null.
- **The decision:** **reject at e ≥ 20** (α = 0.05). This is one primary
  hypothesis, so no multiplicity charge applies.

  ```
  python3 scripts/ev_window.py --label win-a-model --odds proboxingodds_v3.parquet \
      --from 2016-06-10 --to 2020-06-10 --real --json results/window_a.json
  ```

**Window B, secondary, decides nothing: 2021-06-11 → 2023-06-10.**

- The model is trained the same way with `--cutoff 2021-06-10`, label
  `win-b-model`.
- Prices are Bet365's opening line (`odds_external/betsapi.parquet`), with the
  same filter and the same e-value. It is reported against the same bar of 20.
- **This window has been seen**, and it is here only because the Bet365 prices
  in it are new. The report fitted the blend weight against the opening price
  on 2020-06 → 2023-06 (λ = 0.37, ProBoxingOdds), and `rule_oos.py` scored a
  cruder betting rule at ProBoxingOdds' open on 2021-06 → 2023-06 and on
  2020-06 → 2023-06. Whatever it shows is not evidence of the kind window A can
  give.

## 3. Order of operations, and what counts as a pass

1. This protocol is hashed, timestamped and pushed.
2. Both models are trained. **Their logs are not read before step 4.**
   market_eval prints scores against the closing price for every priced bout
   after its cutoff, and those include the windows.
3. Validation, before any outcome is read. Run
   `ev_window.py … --null 1000` on each window: outcomes are drawn from q
   itself, with prices and stakes held fixed. It passes if the mean e-value is
   within Monte Carlo error of 1 or below, and no more than 5% of runs reach 20.
   If it fails, nothing is scored until the instrument is fixed, and the fix is
   a deviation.
4. One real run per window, with `--real`.

Reported whatever the result, and deciding nothing:
- the fair-odds e-value;
- the quiet filter at 0, 30 and 90 days;
- each calendar year;
- flat 1-unit stakes at every λ on the grid, with bootstrap intervals;
- log-loss of the model, the opening price, and the two blended.

## 4. What was seen before this was written

- **Window A.** No script or document in the repository scores the model
  against an opening price on bouts before 2020-06-10. That was checked by
  searching every cutoff and window the scripts and documents name. But the
  window is not untouched in other ways:
  - its bouts are in the training data of the published model;
  - the project's feature choices were made on corpus holdouts, which read
    outcomes but no prices;
  - ProBoxingOdds' closes from 2016–2023 chose the de-vig method, through its
    calibration slope on the quoted training slice.
  - The strategy under test was found on 2023–2025. This is its replication
    on earlier bouts, not a first look.
- **Window B.** Section 2.
- **The scorer's null mode** was run once, 20 replications, on ProBoxingOdds'
  prices for 2023-06 → 2025-12 with the existing open-safe model, to check that
  it runs. That window's outcomes were not read.

## 5. Limitations, stated first

- ProBoxingOdds' "open" has no timestamp. The 60-day filter bounds the risk of
  a bout between open and bell, but does not remove it for lines opened earlier
  than that.
- 2016–2020 is a different market from 2023–2025: fewer books, and less money
  on boxing. A failure here could mean the edge is recent, not that it was never
  there. A pass could mean the market was softer then. Either way it is one
  more window, not the future.
- The only test that nothing historical can contaminate is a forward one:
  predictions published before the bell, graded afterwards.
