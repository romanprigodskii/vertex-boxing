# Model — the boxing port

Adversarially-verified plan (2026-07). The vertexmma model is honest and
mature; this documents exactly what carries over and what must change. The plan
below is kept as written; what follows immediately is what actually happened
when it was run.

## Where it stands (measured 2026-08-01, second pass)

One command reproduces it: `market_eval.py --tta --mirror --blend`. The
defaults ARE the best known model. Three instruments, because the quoted set alone cannot resolve
0.005 and the corpus alone is four fifths club boxing the market never prices.

| instrument | n | log-loss | was | paired gain |
|---|---|---|---|---|
| corpus holdout | 89,087 | **0.3325** | 0.3390 | +0.0065 |
| premium holdout (sched ≥8, both ≥8 bouts) | 12,536 | **0.2845** | 0.2901 | +0.0056 |
| quoted, against the close | 3,288 | **0.3725** vs the market's 0.3469 | 0.3799 | +0.0074 |

The gap to the closing line is **−0.0255**, down from −0.0329: a fifth of it
closed. The blend now beats the close by **+0.0045 [+0.0024, +0.0066]** (λ 0.18,
up from 0.15 — the price has less to say about the model than it used to), and
clean CLV rose from +0.0106 to **+0.0114 of probability [+0.0092, +0.0135] over
1,979 bets** at a 2% edge.

Walk-forward retraining every 12 months is worth a further +0.0024 on the
confirmation half and +0.0014 on the premium holdout, five seeds, and it is not
a flag on the scoreboard: retraining is a property of a deployment, not of a
measurement. `lab.py --exp walk-tta` has it.

**It stacks with the mirror almost exactly.** On top of mirror training it is
worth +0.0023 on the confirmation half [+0.0015, +0.0030] and +0.0014 on the
premium holdout [+0.0002, +0.0026] — against +0.0024 and +0.0014 on top of plain
`--tta`. The two contributions are independent, which is what the mechanisms
predict: the mirror removes an asymmetry between the corners, walk-forward
removes staleness, and neither touches the other. The full stack, measured on
the bench (`lab.py --exp mirror-walk`, three seeds):

| instrument | full stack | scoreboard (`--tta --mirror`) | where it started |
|---|---|---|---|
| corpus holdout | **0.3310** | 0.3325 | 0.3390 |
| premium holdout | **0.2833** | 0.2845 | 0.2901 |
| quoted vs the close | **0.3719** | 0.3725 | 0.3799 |
| gap to the close | **−0.0250** | −0.0255 | −0.0329 |

Three seeds reproduce five here to a thousandth — mirror training scored 0.3325
corpus and 0.3279 confirmation on both — so the cheaper protocol was not a
compromise on this question.

### What the second pass changed, in order of size
1. **87 new features** (`everyx`, 200 columns), +0.0025 on the confirmation half
   [+0.0017,+0.0033]. Not one of the six groups is worth anything alone; the
   block is. What carries it is `cmp` (+0.0016) — do these two ratings even
   come from the same graph — and `thin` (+0.0006).
2. **Test-time symmetrisation** (`--tta`), a further +0.0017 on the confirmation
   half and +0.0030 on the quoted set. A boxing match has no corner A, but the
   model's answer depended on which name was typed first; asking it both ways
   and averaging the two logits cancels the half of that which is noise. One
   extra forward pass, no retraining.
3. **Training on both orientations** (`--mirror`), a further **+0.0022 on the
   confirmation half [+0.0017,+0.0028] and +0.0020 on the premium holdout
   [+0.0009,+0.0032]**, five seeds, at 2.5× the training time.
4. **Retraining as time passes** (walk-forward, 12 months), **+0.0024 on the
   confirmation half and +0.0014 on the premium holdout**, five seeds — and
   +0.0030 on the selection half against +0.0043 on the confirmation half,
   which is the signature of staleness rather than of a better model, since the
   selection half is the year right after the cutoff and the confirmation half
   is two years later. Deployment would do this anyway.

### The three defects the mirror found
Building the mirrored matrix by a second replay rather than by negating columns
made a test possible that had never been run: every column must be a difference
that negates, a quantity invariant to the swap, a probability that becomes 1−p,
or half of a pair that trades with its twin. `mirror_check.py` runs it and
found three columns that failed — `city_home_bias`, `jud_fav` and the
`d_elo_jud` built on it. All three were tie-breaks written `>=`, so when two
fighters had equal ratings (or neither was local) the update fell through to
"corner A" and a quantity that is a fact about a city or a judge came out
different depending on which way round the bout had been written down. Fixed by
refusing to update when there is no favourite and no local man.

### The board has three prices and they answer different questions

| reading | margin | market | model | gap | λ | blend over market |
|---|---|---|---|---|---|---|
| worst price on the board | 6.2% | **0.3469** | 0.3725 | −0.0255 | 0.18 | +0.0045 [+0.0024,+0.0066] |
| best price on the board | 3.0% | 0.3527 | 0.3774 | −0.0247 | 0.19 | +0.0044 [+0.0019,+0.0068] |
| opening line | 5.6% | 0.3591 | 0.3729 | −0.0138 | 0.37 | +0.0096 [+0.0053,+0.0139] |

(was −0.0329 / −0.0306 / −0.0208 at λ 0.15 / 0.16 / 0.33 before the second pass;
λ rising is the point — the price now accounts for less of what the model knows.
The first row is `--tta --mirror`; the other two are `--tta` only and have not
been re-run with the mirror, which is worth about 0.0005 to the model column.)

**The best price is the worst forecast.** Best-of-market is not the market's
opinion; it is the upper envelope over ten books, so taking the maximum on both
sides picks out the two books that most disagree with consensus. Bet at the best
price, score against a balanced one. The blend beat the market under all four
de-vig methods on the best price before the second pass (power +0.0040, Shin
+0.0046, additive +0.0045, proportional +0.0062) and the second pass moved the
power reading, the one `--devig auto` picks, to +0.0044. The other three have
not been re-measured on the new model; they were never the binding case.

### The margin defence is closed: a price with nothing to remove

Every other number in this file is scored against a bookmaker, and a bookmaker's
price has a margin in it that has to be modelled away first — so there is always
a lingering "maybe the model only trails a particular reading of the de-vig".
Polymarket settles that. It is a two-sided traded book with essentially no
margin, and its history is recoverable from Polygon (`run_polymarket.py`:
84,729 fills across 67 who-wins markets, $37.9M of volume, 2024-01 to 2026-08).

`polymarket_eval.py` matched 14 of them to corpus bouts. On the 12 where the
bookmaker also priced the fight:

| | log-loss |
|---|---|
| bookmaker close, power de-vig | **0.3274** |
| Polymarket, no margin at all | 0.3643 |
| our model | 0.4402 |

**The model trails a zero-margin price, so the margin was never the explanation.**
And the price used here is deliberately handicapped: it is the median of the last
25 fills strictly before 00:00 UTC on the day of the bout, so it throws away the
day-of movement and is duller than the real close. The conclusion holds a
fortiori.

Two things fall out of it that are worth more than the headline.

**The de-vig is audited and it passes.** The zero-margin price and our
power-de-vigged bookmaker close agree to a median of 0.047 of probability. Two
independent sources, one with the margin removed by our method and one with no
margin to remove, land in the same place. That retires the worry the whole
de-vig section was written to manage. It also serves as an orientation check on
this join: had a corner been flipped anywhere, the gap would sit near 0.5 and
Polymarket would score worse than a coin.

**Our recorded opening line is genuinely early, not a settled number.** The
worry was that `open_*` is whatever proboxingodds happened to write down, which
might be hours or days after the market actually opened — in which case the real
opener would be softer and every edge calculation understates us. The chain
answers it: Polymarket's FIRST fills sit a median of 0.058 from the feed's open
and 0.049 from the feed's *close*, so if anything the feed's open is the earlier
of the two numbers. As a forecast the feed's open scores 0.3318 against
Polymarket's 0.3596 at its own first fills. There is no hidden softness to catch.

That kills one reason to build forward capture and leaves the real one standing:
history cannot be bet. The only way to test an edge with money is to take prices
going forward.

The caveat matters more than the result. These are twelve marquee bouts — Usyk,
Canelo, Fury — the most efficiently priced segment of the sport. **The thesis
this project exists to test is about soft REGIONAL lines, and nothing here
touches them**: proboxingodds prices mostly the notable layer and Polymarket
prices only stars.

**The bookmaker beat the prediction market on these fights** — 0.3274 against
0.3643 — which is not what the "near-zero-margin markets are sharper" story
predicts. On twelve bouts that is noise and nothing more, and it is recorded
only so that nobody rebuilds the yardstick around Polymarket expecting it to be
strictly better. The model−Polymarket gap over all 14 is +0.0682 with an
interval of [−0.0668, +0.2023]: this instrument cannot resolve anything near
0.002, and was never going to.

### The two results that are not circular
The model never sees a price, so its closing-line value is clean: taking its
picks at the OPEN and marking to the close is worth **+0.0114 of probability
[+0.0092, +0.0135] over 1,979 bets** at a 2% edge (was +0.0106 before the second
pass; +0.0152 at a 5% edge over 1,323). Flat-staked ROI into the real closing
number also improved at every threshold — −6.30% → −3.25% at a 2% edge,
+5.10% → +6.41% on competitive bouts — though none of those intervals excludes
zero on this sample.

Measure CLV in PROBABILITY. The price-space null — bet a random side at the open,
mark to the close — is +3.30% reading the close as the worst price and −4.9%
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

The six groups added on 2026-08-01, leave-one-out on the corpus holdout at one
seed (so ±0.001, and only `cmp` is clear of it):

| group | what it is | dropping it costs |
|---|---|---|
| `cmp` | same passport, venue-histogram cosine, days since a stoppage loss, journeyman index, rating z-score against the population active that month | **+0.0016** |
| `lvlr`+`lvlq` | the LEVEL of every rating and rate, not only the difference — min and max over the two corners | +0.0010 |
| `thin` | one-sided-tolerant records, and what the card says when one man has none | +0.0006 |
| `res` | performance against what Elo expected, and the fall from a career peak | +0.0006 |
| `ctx` | running upset rate by country, promoter, division, distance | +0.0005 |
| `unc` | Glickman's expected score, gaps divided by their own standard error | +0.0000 |

Read that table as a whole, not row by row: none of the six is worth anything on
its own (each scored +0.0000 to −0.0002 when added to `everyc` alone), and
together they are worth +0.0025. They are six ways of saying the same thing —
how much should a rating gap be trusted HERE — and the model needed enough of
them at once to tell the cases apart.

### The three instruments, and why a one-seed screen cannot see 0.001
The corpus holdout is split in half by date: the EARLY half selects, the LATE
half reports, and nothing is ever chosen on the late half (`lab.py`). And seed
noise is not bout noise — a paired bootstrap over bouts does not see it at all.
The same configuration on three seed sets scores 0.3375 / 0.3377 / 0.3379 on the
corpus and 0.3326 / 0.3332 / 0.3336 on the confirmation half, so **a one-seed
screen resolves about 0.001 and no better**. Everything above 0.002 was
re-measured on five seeds before it was believed; everything at 0.001 is
reported as one seed and labelled as such.

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

Added 2026-08-01, both of them answers to "surely the weights should differ on
the fights that are not close":

**A λ that varies with how lopsided the PRICE is** (`--bandblend`). This is not
the fading blend that failed — that one keyed λ on the model-market
disagreement, a quantity the blend partly creates; this keys it on the price's
own distance from even, which is fixed before the model speaks. It still loses:
0.3434 against the constant blend's 0.3428, +0.0035 over the market instead of
+0.0041. The fitted weights say why — 0.13 / 0.16 / 0.27 / 0.13 across the four
bands, non-monotone, with the most competitive band fitted on 260 bouts. There
are 2,472 bouts in the fitting window and four parameters do not survive being
carved out of them. The constant λ keeps winning for the same reason it beat the
two-weight logistic and the fade: one parameter is all this window can pay for.

**Weighting training bouts by how close the ratings said they were**, 4p(1−p)
from the Elo-implied probability with a floor, so that the nine tenths of the
corpus which is a padded prospect against a journeyman stops dominating. Again
not the quoted-population reweighting that failed, which asked "is this the kind
of card the market prices" rather than "was this fight in doubt". Floors 0.50 /
0.25 / 0.10 give corpus 0.3350 / 0.3350 / 0.3343 against 0.3348 for no weighting
— the most aggressive is worth about +0.0007 on the confirmation half, which is
two standard errors at three seeds, and it is worth **nothing at all** on the
premium holdout (0.2860 either way) or the quoted set (0.3729 against 0.3727).
A gain that appears only on the four-round club boxing the market never prices
is not a gain worth having.

**Averaging models fitted with different half-lives** (3, 6 and 12 years, logits
averaged). The idea is sound — the members disagree for a reason rather than by
accident, which is what usually makes an average beat its best member — and it
returns +0.0003 on the confirmation half and +0.0006 on the premium holdout for
three times the training cost. Half-life is simply not a live axis here: 6 years
was already close enough to the optimum that spreading around it buys nothing.

### The diagnosis, rewritten from the second pass
The old diagnosis — "the model is over-confident where the price says pick'em" —
was reading a per-bout table without weighting it by how many bouts are in each
band. Weighted, the gap is nearly FLAT across the price scale (+0.0066, +0.0073,
+0.0043, +0.0070, +0.0072 of the total, from 95% favourites down to coin flips).
And the model is not miscalibrated: its slope is 0.989 on the corpus holdout,
0.998 on the premium one, 0.943 on the quoted set. Shrinking every logit by a
factor chosen *on the test set itself* — an oracle no honest model gets — buys
0.0005. **There is no calibration fix, because there is nothing to calibrate.**

What there is instead, from `where.py`:

- **77% of the gap is on bouts where the MARKET is bolder than we are**, not the
  other way round (0.3996 against 0.3449 there, against 0.3628/0.3487 where we
  are bolder). The failure is missing information, not misplaced confidence.
- **Where our own evidence runs out, it runs out completely.** On the 205 bouts
  (6.2%) where the less experienced man has fewer than three recorded bouts, the
  market scores 0.1379 and we score 0.2491 — **19.7% of the whole gap**. A
  debutant priced at 97% is an amateur international and the corpus has never
  heard of him.
- **Where we say "coin flip" the market usually is not guessing**: on the 373
  bouts where our |logit| is under 0.5, the market scores 0.5986 to our 0.6889 —
  31% of the gap.

### Amateur pedigree: pulled, matched, measured, and it does not work
This was the top of the "what next" list and it is now a dead end, which is
worth more written down than guessed at.

BoxRec was the obvious source and is the wrong one: zero of 501 saved
professional profile pages contain the word "amateur", the amateur records live
in a separate id space with no link from the pro page, and reaching them means
logged-in search at roughly 150 pages a day. Wikidata has the same men for free
(`13_wikidata_amateur.py`): 6,733 dated appearances at the Olympics and the
major games, of which 6,090 matched 4,804 of our fighters by the BoxRec
crosswalk and the Wikidata id (`14_ingest_amateur.py`). Only appearances DATED
BEFORE a bout are ever visible to it, so a 2012 fight cannot know about a 2016
medal.

The raw signal is strong. On the quoted test set, when exactly one of the two
men has an amateur international behind him he wins **73.4% of 458 bouts**, and
**82.3% of 141** when he medalled.

None of it is incremental, and adding it makes the model worse:

| premium holdout | log-loss |
|---|---|
| without the group (200 features) | **0.2874** |
| plus its single strongest column | 0.2880 |
| plus all six (206 features) | 0.2889 |

Monotone in how much of it you add, and the six-feature version costs +0.0014
[+0.0004, +0.0025] on three seeds. The damage is concentrated exactly where the
features fire — +0.0106 on the quoted bouts with a pedigree, +0.0168 on the
premium ones — which names the mechanism: the group is non-zero on 1.2% of
training rows and on 14.7% of the quoted test set, so the model learns the
effect from club boxing where an Olympian is a rarity and then applies it to a
population where they are common. The ratings, the records and the level
features already carry everything it knows.

It stays computed and out of the default set: `--feats everyx+amat` reproduces
the negative.

### What would move the needle next
Amateur pedigree was item one on this list and has been struck off above, which
leaves the thin-record slice still open and no cheap source for it: what the
market knows about a debutant is not that he was an Olympian — we now have that
and it does not help — but what his camp, his gym and his last sparring looked
like, and none of that is in any database.

1. **Walk-forward in production.** +0.0019 on the second half of the holdout for
   nothing but a cron entry, and it grows with the age of the model.
2. **The yardstick.** proboxingodds.com's front page is a LIVE ten-book board
   including Polymarket and Kalshi — near-zero-margin prediction markets, a far
   sharper benchmark than any historical close. Forward capture is at zero of
   five parts.

   **Correction, 2026-08-01: "nothing about it can be recovered after the fact"
   was wrong, and it was wrong in the way that matters.** That is true of the
   aggregated proboxingodds view, and it is NOT true of Polymarket, which is
   the sharpest book on that board. Polymarket settles on Polygon, so every
   trade is permanent public record. `prices-history` on the CLOB API returns
   empty for a closed market and 400s on an explicit range, which is what makes
   it look unrecoverable — but `data-api.polymarket.com/trades` paginates
   through the whole on-chain log. Usyk–Fury 2 comes back complete: 6,971
   trades from 2024-12-03, 3,004 of them before the bell, and the mean of the
   last twenty is 0.506 for Usyk, who then won on points. That is a closing
   line with no margin in it at all, reconstructed two years after the fact.

   What it is worth is a separate question from whether it exists. There are
   214 closed boxing events, 64 above $50k of volume, $198M in total — but the
   volume is influencer boxing: Paul–Tyson and Paul–Joshua are $115M of it, and
   the rest of the top twenty is crypto fight nights and prop markets on the
   method. Strip the props, strip the celebrity cards, and deduplicate the two
   framings of the same bout ("will Usyk beat Fury" and "will Fury beat Usyk"),
   and the sanctioned-boxing set inside our test window is on the order of ten
   to twenty-five fights. That cannot resolve 0.002 — the interval on twenty
   bouts is wider than the entire gap being argued about.

   It is still worth having, for a reason that is not statistical power. A
   Polymarket price carries essentially no margin, so it settles the question
   the de-vig can only argue about: on the fights where it exists, is the model
   behind a *fair* consensus, or only behind a bookmaker's? And the same two
   endpoints are the forward-capture pipeline, which means that part of the
   plan now has a concrete target instead of a scraper against a front page.
3. Not hyper-parameters. Re-checked on the real protocol: 63→31→127→255 leaves,
   lr 0.03→0.015, min_data 30→300, λ₂ 5→30, feature fraction 0.9→0.6 — the best
   of them is worth +0.0006 on the confirmation half and none has an interval
   clear of zero. And not the quoted-population reweighting (−0.0019), nor
   folding draws into the target (+0.0000), nor Glickman's expected score as a
   boosting offset (+0.0001).

### The four scripts that hold the method
| script | what it is for |
|---|---|
| `market_eval.py` | the scoreboard. One command, three instruments, the blend |
| `lab.py` | the bench. One data load, many variants, a paired verdict each; the holdout split into a half that selects and a half that reports |
| `mirror_check.py` | every column of the mirrored matrix must negate, stay put, become 1−p, or trade with its twin. Found three real defects the first time it ran |
| `leak_check.py` | no feature's missingness may name a corner |
| `where.py` | where the market's advantage sits on axes we can see before the bell |

## The thesis points the wrong way

This project exists on one claim: that regional and club lines are soft because
nobody sharp is betting a six-rounder, and that is where the money is. Every
number above is an average over the quoted set, which cannot speak to it — an
average hides exactly the thing the claim is about. `regional.py` splits the
quoted test bouts by level and looks inside. The betting threshold is fixed in
advance at 2% on the opening price, and CLV is the column to read, because ROI
on a few hundred bets is noise and CLV converges about twenty times faster.

Every proxy for "how big was this fight" gives the same answer, and it is not
the answer the thesis predicts.

| by scheduled distance | n | gap to close | CLV | ROI |
|---|---|---|---|---|
| 4–6 rounds (club) | 597 | +0.0428 | +0.0043 [+0.0014,+0.0074] | −20.2% |
| 8 rounds (regional) | 582 | +0.0042 | +0.0091 [+0.0048,+0.0137] | +2.9% |
| 10 rounds | 1,195 | +0.0325 | +0.0097 [+0.0063,+0.0131] | +4.0% |
| 12 rounds (title) | 629 | +0.0242 | **+0.0183** [+0.0129,+0.0236] | **+14.6%** |

| by belt and card size | n | gap to close | CLV | ROI |
|---|---|---|---|---|
| no belt, card < 8 bouts | 707 | +0.0256 | +0.0089 | +4.1% |
| no belt, card ≥ 8 | 1,239 | +0.0270 | +0.0060 | −10.2% |
| regional or national belt | 378 | +0.0294 | +0.0150 | +8.4% |
| continental/world belt | 964 | +0.0221 | **+0.0156** | **+12.4%** |

By how much record the two men carry it is the same shape: +0.0085 under eight
bouts, +0.0125 at 8–15, +0.0120 at 15–25.

**The edge is at the top of the sport, not the bottom.** Closing-line value more
than quadruples from club fights to title fights, and flat-staked ROI goes from
−20% to +15%. Three proxies that measure different things — the distance, the
belt, the depth of the records — agree.

The mechanism is the one already diagnosed from the other end. Our features are
records and ratings, and a rating is only as good as the record under it. At
world level both men have thirty fights and the ratings are tight; on a club
card the market knows which of these two is a prospect being moved and we are
reading two thin records. It is the thin-record finding again — 19.7% of the gap
on 6% of the bouts — seen from the level axis instead of the record axis.

### What the finding prescribes

Same model, same predictions, same 2% threshold on the opening price — only a
filter on the level of the bout, which is knowable before the bell and has
nothing to do with the price.

| filter | bouts | bets | CLV | ROI |
|---|---|---|---|---|
| all quoted | 3,288 | 2,037 | +0.0109 [+0.0088,+0.0130] | +2.7% [−3.1%,+8.7%] |
| 10+ rounds or any belt | 2,033 | 1,378 | +0.0128 [+0.0100,+0.0157] | +6.2% [−1.1%,+13.7%] |
| 12 rounds or a continental/world belt | 1,134 | 798 | **+0.0148** [+0.0111,+0.0188] | **+11.3% [+1.8%,+21.2%]** |

Both columns rise monotonically as the filter tightens, and the last row is the
first flat-staked ROI interval on this model to exclude zero.

**Read the selection caveat before believing the last cell.** The cut-points
were chosen after looking at the level table, so the +11.3% is a post-hoc
number and its interval understates the real uncertainty. What is NOT post-hoc:
the direction was agreed by three proxies measuring different things before any
cut was picked, CLV is monotone across every band rather than only at the chosen
one, and the filter is a coarsening of an existing pre-bell feature rather than
a parameter fitted to the outcome. The honest status is "a strong hypothesis
with a mechanism", and the thing that would settle it is out-of-sample
confirmation — which is what forward capture is for.

**What this does and does not license.** It does not prove the unpriced regional
tail is efficient, and cannot: a four-rounder that gets a line at all is not the
anonymous tail, it is a prospect showcase on a televised undercard, which is the
one part of club boxing the market watches closely. The genuinely unpriced
segment stays unmeasured because nothing prices it.

What it does say is that the plan of chasing the regional tail has no support in
any evidence this project has ever produced, and the evidence that exists points
the other way. Effort should go where the edge is measurable: deep records, high
level, and the fights where our ratings are actually estimated.

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
