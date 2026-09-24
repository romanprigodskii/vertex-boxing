# Pre-registration 3: which bouts to bet at the open, chosen first, tested once

Roman Prigodskii · written 2026-09-24. This file and `ev_rules.py` are hashed into
`docs/evalue_protocol_3.manifest`. That manifest is timestamped with
OpenTimestamps and pushed to the public repository before any candidate below is
scored.

## 1. Why

Pre-registration 2 found that betting at the opening price, with the model
blended into it, made money on 2016–2020 and on 2023–2025, and not on 2021–2023.
It bet every bout. The report had already shown where the model adds least: on
club bouts, whose blend weight is zero (REPORT 4.2). So the obvious next rule
leaves those bouts out. Choosing that rule by looking at the windows already
used, and then quoting its result on the same windows, would be fitting, not
testing. So the sets are named first.

| set | period | used for |
|---|---|---|
| training | before each window's cutoff | the model's weights |
| **selection** | windows A (2016-06 → 2020-06), B (2021-06 → 2023-06) and C (2023-06 → 2025-12), each with its own model | choosing the rule |
| **holdout** | 2026-01-01 → 2026-07-24 | one run of the chosen rule |
| live | from the day a rule is published | the only final answer |

## 2. The candidates

Every candidate bets the same way as pre-registration 2:
- q is the opening price, the margin removed by the power method;
- the alternative is the blend σ(λ·logit p + (1−λ)·logit q), mixed uniformly
  over λ ∈ {0.05, …, 0.50};
- the stake is Kelly at the posted opening decimals;
- bouts where either fighter boxed in the 60 days before are left out.

A candidate only decides which bouts are bet.

| rule | bets on |
|---|---|
| R0 | every bout (the rule of pre-registration 2) |
| R1 | bouts scheduled for 8 rounds or more: no club bouts |
| R2 | title bouts: any belt on the line, or scheduled for 12 |
| R3 | the report's upper tier: 12 rounds, or a continental, international or world belt |

## 3. The choice

`ev_rules.py dev` scores each candidate at real opening prices on windows A, B
and C, each with the model trained before it:

| window | model | prices |
|---|---|---|
| A | `win-a-model` | ProBoxingOdds v3 |
| B | `win-b-model` | Bet365 |
| C | `openinfo-close` | Bet365 |

It sums the natural log of the e-value over the three windows. **The candidate
with the largest sum is chosen**, and is written to `results/rules_dev.json`.
That rewards a rule both for how fast it grows and for how many bouts it keeps,
because a holdout of this size is a test of both. The file is committed and
pushed before the holdout model is trained.

## 4. The holdout

- **Model:** trained on the corpus up to 2025-12-31, in the configuration of
  pre-registration 2, without fight-week features:
  ```
  VERTEX_ODDS=<repo>/imports/staging/proboxingodds_v3.parquet \
  python3 scripts/market_eval.py --tag l6 --tta --mirror --xt --seeds 3 \
      --cutoff 2025-12-31 --drop weigh+ref+judc+cardpos --label hold-model
  ```
  Its log is not read before the real run, because it scores the model against
  the closing price on the holdout.
- **Prices:** Bet365's opening line, the file whose digest is in
  `data_manifest.json`.
- **Bouts:** 2026-01-01 → 2026-07-24, quiet by the 60-day rule. By dates alone
  there are 541: R0 541, R1 404, R2 243, R3 192.
- **Validation first:** `ev_rules.py holdout --null 1000` on the chosen rule.
  It passes if the mean e is at most 1 within Monte Carlo error and no more
  than 5% of runs reach 20.
- **The decision:** one real run of the chosen rule. **Reject at e ≥ 20.** The
  other three candidates are reported beside it and decide nothing.

## 5. What was seen before this was written

- **The selection windows have all been scored for R0,** in pre-registration 2
  and in the follow-up to the first audit. R1–R3 have not been scored on them
  at real opening prices. But the report's closing-line-value table by level,
  and λ by slice on the fitting window, were known when the candidates were
  written.
- **The holdout's bouts were scored once, on 2026-09-23, against Bet365's
  closing price, with the published model:**
  - in aggregate, and by scheduled distance (club bouts −0.0446, 12 rounds
    −0.0088);
  - the published λ = 0.17 blend at the close.

  Nothing at the opening price, and no betting, was scored on them. The
  holdout model does not exist yet.
- **The counts in section 4** come from dates and covariates alone.

## 6. What it can and cannot show

541 bouts is a small holdout. If the effect were the size it was on 2023–2025,
R0 would expect a log e-value of roughly 3 to 4, so e of 20 to 50. Narrower
rules keep fewer bouts. So a pass says the chosen rule held on one more half-year
that nobody had seen. A failure does not refute it. The live test in section 1
is still the only final answer.
