# Pre-registration 4: the live test

Roman Prigodskii · written 2026-09-24. This file, the scorer, the feature code
and the training code are hashed into `docs/evalue_protocol_4.manifest`, beside
the digest of the frozen training corpus. The manifest is timestamped with
OpenTimestamps and pushed to the public repository. **The bouts this test is
scored on had not happened when it was pushed.**

## 1. Why this one is different

Every earlier test in this repository scored outcomes that already existed. The
best any of them could do was to keep the analyst from seeing those outcomes
before the rules were fixed, and each says how far that succeeded. This test
fixes the rule before the outcomes exist, so there is nothing to have seen.

It is needed because the rule it tests was chosen after looking. The blend weight
λ = 0.10 was the strongest of ten on the 2026 holdout of pre-registration 3,
after that holdout had been scored. Flat stakes at that weight were positive in
every window before it, but that is a pattern found by looking, not a test.

## 2. The rule, frozen

- **The model.** Trained on the frozen corpus up to its last date, 2026-07-24,
  in the configuration of pre-registrations 2 and 3, without fight-week
  features:

  ```
  VERTEX_ODDS=<repo>/imports/staging/proboxingodds_v3.parquet \
  python3 scripts/market_eval.py --tag <l6 extended> --tta --mirror --xt --seeds 3 \
      --cutoff 2026-07-24 --drop weigh+ref+judc+cardpos --label live-model
  ```

  The training rows are the `l6` snapshot, whose SHA-256 is in the manifest.
  Bouts after 2026-07-24 are appended to it for scoring and never trained on.
  The replay is point-in-time and training is deterministic. So the code hashed
  here plus that snapshot is the model: whoever runs the command gets the same
  one.
- **The price.** Bet365's opening line, from BetsAPI's archive, which records
  the time of every price itself. q is the price for corner A with the margin
  removed by the power method.
- **The bet.** p′ = σ(0.10·logit p + 0.90·logit q). A side is bet, 1 unit, where
  p′ times its posted opening decimal is above 1.
- **The bouts.** Every bout dated after 2026-09-25 with a Bet365 opening price.
  A bout is left out if either fighter boxed between Bet365's opening timestamp
  and the bell, because the model would know that result and the price would
  not.

## 3. The test

Each bet returns r = d − 1 if it wins and −1 if it loses. **H0: every bet the
rule places has an expected return of at most zero at the decimal Bet365
posted.** Staking a fraction c of wealth on each bet in turn multiplies wealth by
1 + c·r, which has expectation at most 1 under H0 for any c in [0, 1]. The
product, mixed uniformly over c ∈ {0.05, 0.10, …, 0.50}, is an e-value, and by
Ville's inequality it may be read at any checkpoint.

**H0 is rejected the first time e ≥ 20 at a checkpoint.** Checkpoints can be taken
as often as data arrives, and looking costs nothing. The test ends at the first
rejection or on 2029-09-25, and the result is reported either way.

The scorer (`ev_forward.py`) was checked before this was pushed:
- **null simulation:** at the edge of H0, where each backed side wins with
  probability exactly 1/d, 2,000 runs give a mean e of 0.99 ± 0.05, and 0.2%
  reach 20;
- **dry run on the 2026 holdout:** that is the data λ was chosen on, so it
  shows only that the code runs and the test has power. There, 39 bets returned
  +37% flat and e = 38. An earlier draft that staked Kelly at λ = 0.10 reached
  only 1.7 on the same bets, because at that weight the blend sits so close to
  the price that Kelly bets almost nothing. So the test is on the rule's own
  flat bets.

Reported at every checkpoint, deciding nothing:
- flat ROI with a bootstrap interval;
- profit in units;
- the Kelly version;
- the λ-mixture of the earlier protocols.

## 4. What a checkpoint needs

1. Bet365's prices after 2026-09-25, from BetsAPI (`backfill_betsapi.py`; a
   three-day token covers months).
2. The corpus extended past 2026-07-24 with the bouts that have happened, for
   the fighters' records and the outcomes.
3. The command in section 2, then `ev_forward.py --null` and `--real`.

None of this has to happen before the bouts. Every input is either frozen here
or recorded by someone else at the time: Bet365's prices, and the fights
themselves.

## 5. Honest expectations

Bet365 prices about 1,400 boxing bouts a year. At λ = 0.10 the rule bets
roughly one in fifteen of those, so something like 60 to 90 bets a year. If the
true return were the +19% the rule averaged over 2016–2026, e would be expected
to reach 20 after roughly 60 to 100 bets, one to two years. At +10% it would take
three to four. At zero it would never get there. A bookmaker limits an account
that wins at the open, so even a pass says the price was beatable, not that it
could be bet at size.
