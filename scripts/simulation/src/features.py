"""One chronological replay, every feature, one place.

There were five copies of replay() with drifting defaults, so two commits'
numbers were never comparable and a fix like "the dates are truncated" had to be
made five times. Experiments are thin wrappers over this now.

Everything here is knowable BEFORE the opening bell, and the line runs through
the middle of the event page rather than around it. The finish — method, round,
the judges' totals — is the one thing that is NOT knowable, so it never becomes
a feature of the bout it came from and may only update a rating for the next
one. But the weigh-in happened yesterday and the officials were assigned before
that, so the scales and the names of the referee and the three judges are on the
card by the time the closing line is set. They are separate groups (WEIGH, OFF)
because they are fair against the CLOSE and not against the open, and a run that
uses them may not be quoted as evidence that we could have bet early.

And "knowable before the bell" is not the same as "not contaminated by the
future". Whether OUR database happens to hold a man's date of birth is decided
by a crawl that ran in 2026 and targeted the fighters the odds feed quotes —
so the shape of what is missing is hindsight, and it leaked the winner. See
AGE below, and scripts/leak_check.py, which makes the test permanent.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STAGING = ROOT / "imports" / "staging"

# bumped whenever the feature matrix changes, so a cached parquet from an older
# definition can never be silently reused under a new one
FEATS_VERSION = 10

ELO_K, ELO_INIT = 32.0, 1500.0
# division → its nominal pound limit; a real ordering beats a category code
DIV_LBS = {"minimumweight": 105, "light_flyweight": 108, "flyweight": 112,
           "super_flyweight": 115, "bantamweight": 118, "super_bantamweight": 122,
           "featherweight": 126, "super_featherweight": 130, "lightweight": 135,
           "super_lightweight": 140, "welterweight": 147, "super_welterweight": 154,
           "middleweight": 160, "super_middleweight": 168, "light_heavyweight": 175,
           "cruiserweight": 200, "heavyweight": 250}
STOP = {"ko", "tko", "rtd"}
# a result that says nothing about who was better on the night
NO_VERDICT = {"dq", "nc"}

# a_*/b_* columns that must move together when the corners are swapped
PAIRED = [("a", "b"), ("a_name", "b_name"),
          ("a_bouts_before", "b_bouts_before"),
          ("a_wins_before", "b_wins_before"),
          ("a_losses_before", "b_losses_before"),
          ("a_draws_before", "b_draws_before"),
          ("a_dob", "b_dob"), ("a_height", "b_height"), ("a_stance", "b_stance"),
          ("a_lbs", "b_lbs"), ("a_score", "b_score"),
          # the judges' individual cards are per-corner too, and they are
          # comma-joined strings, so swapping the strings swaps the corners
          ("judge_a", "judge_b"), ("a_ctry", "b_ctry")]

BASE = ["d_elo", "d_bouts", "d_wr", "d_layoff", "n_a", "n_b",
        "d_home", "d_promo_ties", "promo_bouts", "city_home_bias",
        "sched_rounds", "d_ko_rate", "d_koed_rate", "d_sos", "d_losses",
        "d_career_days", "weight_lbs", "d_form3", "d_sos3", "d_momentum",
        "lay_a", "lay_b", "d_since_win", "d_activity", "both_rusty"]
# what BoxRec printed on the night — covers the career before our corpus starts
RECORD = ["d_bouts_true", "n_true_a", "n_true_b", "d_losses_true", "d_wr_true",
          "a_unbeaten", "b_unbeaten", "d_hidden"]
# quality of opposition, one level deeper than the mean opponent rating
SOS2 = ["d_sos2", "d_opp_losing_share", "d_opp_debut_share"]
# Glicko-2: the rating deviation says "this rating has not converged", which is
# exactly the thing plain Elo cannot express about a 6-bout regional fighter
GLICKO = ["d_glicko", "rd_a", "rd_b", "d_glicko_cons"]
# AGE. age_a and age_b used to be NaN independently of each other, and that was
# a leak, not a gap: the profile crawl went after the ~3k fighters the odds feed
# quotes, those men are listed in the first corner, and the first corner wins
# 87% of the time. P(A wins | only a_dob known) = 0.86 against 0.14 the other
# way, on 68k bouts. So age is now all-or-nothing: unless BOTH birthdays are
# known the whole group is NaN together, which carries no corner information.
# (stance is 100% 'unknown' in the corpus, so stance_clash is dropped.)
AGE = ["d_age", "age_a", "age_b", "d_height"]
LEVEL = ["sched_known", "d_weight_jump"]

# ---- groups added after the leak fix, each measurable on its own -------------
# who has actually shared a ring with whom: rematches, and the transitive
# comparison that boxing's matchmaking makes unusually informative
H2H = ["h2h_n", "h2h_score", "common_n", "d_common"]
# power and chin — the closest thing to the punch stats we will never have
DUR = ["d_never_stopped", "d_dist_rate", "d_rpb", "d_mileage", "d_mile365",
       "d_ko_early", "d_ko_adj", "d_koed_adj"]
# form with a memory that fades, and activity on two horizons
FORM = ["d_streak", "d_wr_dec", "d_b365", "d_b90", "d_dsl", "d_last"]
# level: what class of bout each man has been in before, and what this one is
LEVEL2 = ["card_size", "d_sched_max", "d_sched_mean", "stepup_a", "stepup_b",
          "d_top_opp", "d_beat_top", "d_opp_wr", "d_opp_gl", "d_div_bouts",
          "d_div_share"]
# Elo is one estimator with one setting; these are the same idea run differently
ELO2 = ["d_elo_mov", "d_elo_slow"]
# a whole-history rating, refitted point-in-time — see bt_ratings()
BT = ["d_bt2", "d_bt8", "btn_a", "btn_b"]
# the symmetric half of what the profile crawl tells us: not WHO we know, which
# leaked, but WHETHER we know both — a notability proxy that names no corner
MISS = ["age_known"]
# What the scales said the day before. Legitimate against the CLOSING line and
# not against the opening one, which is why it is its own group: a run that
# uses it may not be quoted as evidence that we could have bet early.
WEIGH = ["d_lbs", "over_a", "over_b", "d_over", "d_lbs_hist", "lbs_known"]
# How much a man wins by, not merely whether. Every rating in this file reads
# one bit per bout — Elo, Glicko and Bradley-Terry all see a win as a win — and
# the judges hand out a graded number on 144,337 of them for nothing.
SCORE = ["d_melo", "d_dom", "d_dom_win", "d_dom_loss", "d_dom3",
         "dom_n_a", "dom_n_b"]
# The third man in the ring and the three at the tables. The referee decides
# WHETHER the fight reaches the cards, which decides which of a fighter's
# qualities matter; the judges decide WHO wins once it does, and the sport's
# own literature puts the home fighter's edge on points at 0.74. Both are
# assigned before the bell. ref_stop_res is the referee's rate net of what
# bouts at that distance produce anyway — without it the feature is mostly a
# label for the kind of card he works, which the model already knows.
REF = ["ref_stop", "ref_early", "ref_stop_res", "ref_n", "ref_home", "d_home_ref"]
JUD = ["jud_home", "d_home_jud", "jud_fav", "d_elo_jud", "jud_n"]
OFF = REF + JUD + ["off_known"]
# What was on the saved event pages all along. Two of these fix things rather
# than add them: the flags give the officials a REAL home fighter instead of
# "the man with more previous bouts in this country", and the position on the
# card is the promoter's own ranking of his show — BoxRec prints it main event
# first, and the first sixth of a card carries 25.7% of the belts and averages
# 8.3 scheduled rounds against 0.1% and 4.3 in the last sixth.
CARD = ["d_home_true", "home_known", "title_lvl", "is_title", "card_pos", "is_main"]

# ---- groups added 2026-08-01 ------------------------------------------------
# LEVELS. Almost every number above reaches the model as a DIFFERENCE, so two
# 1700s and a 1700 against a 1300 are the same row. That is exactly the
# distinction the market makes and we do not: a 200-point edge between two
# world-class men is a fight, and between two novices it is a formality. min and
# max are the orientation-invariant way to say it — a mean would say it too, but
# the min is the one that carries "the weaker man is a total novice".
LVLR = ["elo_min", "elo_max", "gl_min", "gl_max", "rd_min", "rd_max",
        "n_min", "n_max", "ntrue_min", "ntrue_max", "career_min", "career_max",
        "schedmax_min", "schedmax_max", "peak_min", "peak_max"]
LVLQ = ["wr_min", "wr_max", "wrtrue_min", "wrtrue_max", "ko_min", "ko_max",
        "koed_min", "koed_max", "dist_min", "dist_max", "sos_min", "sos_max",
        "oppwr_min", "oppwr_max", "oppgl_min", "oppgl_max", "b365_min",
        "b365_max", "mile_min", "mile_max", "opp3_min", "opp3_max",
        "bt8_min", "bt8_max"]
# The interactions a tree has to discover from scratch and mostly does not,
# because 90% of its training rows are club bouts where they do not bind.
# glicko_e is Glickman's own expected score: the same rating gap, deflated by
# how unconverged the two ratings are. A tree given the gap and the two RDs
# separately has to carve that surface out of axis-parallel cuts.
UNC = ["glicko_e", "d_glicko_z", "d_elo_z", "d_bt_z", "d_elo_less_sos",
       "rating_disagree", "d_elo_x_lvl", "d_glicko_x_sched"]
# Performance against expectation, rather than performance. A man who is 12-0
# against nobody and a man who is 9-3 against contenders have the same shape in
# every rate above; the residual to what Elo expected on the night separates
# them. And the fall from a career peak, which no online rating can express.
RES = ["d_surp5", "d_surp_all", "surp5_min", "surp5_max",
       "d_peak_elo", "d_peak_gap", "d_worst_loss"]
# How predictable is boxing HERE. Running, point-in-time upset rates by country,
# promoter, division and distance — the model's own answer to "how much should a
# rating gap be trusted on this kind of card", read off past bouts of that kind
# and never off the price. home_fav is invariant on purpose: both of its factors
# flip with the corners, so what survives is "the favourite is/is not at home".
CTX = ["ctry_upset", "ctry_mae", "ctry_n", "promo_upset", "div_upset",
       "sched_upset", "home_fav", "p_stop_hat", "d_elo_x_stop"]
# COMPARABILITY and RECENCY. Every rating here is fitted on ONE graph covering
# all of world boxing, but the graph is not connected: a Mexican club scene and
# a British one exchange almost no results, so a 200-point gap across them is
# not the same number as a 200-point gap inside one. same_ctry and venue_cos
# say how comparable the two ratings are; elo_z_pop says where a rating sits
# against the population actually active that month, which is what makes a
# level meaningful across eras. And a man stopped last year is a different
# fighter from the same record without that line in it.
CMP = ["same_ctry", "venue_cos", "d_ko_loss_days", "koloss_min", "koloss_max",
       "jm_min", "jm_max", "hidden_min", "hidden_max",
       "elo_z_min", "elo_z_max", "d_elo_x_ctry", "d_gl_x_title"]
# THIN RECORDS, where a fifth of the gap to the closing line lives. On the 3,288
# quoted test bouts the market scores 0.1379 against our 0.2491 when the less
# experienced man has fewer than three recorded bouts — 6% of the fights and 20%
# of the whole deficit — because a debutant priced at 97% is an amateur
# international and the corpus has never heard of him. What IS knowable is who
# they put him in with and on what kind of card, and the _mm rule was throwing
# exactly that away: one missing record NaN'd the pair, so the opponent's record
# vanished in the only case where it was all we had.
THIN = ["wrtrue_seen_min", "wrtrue_seen_max", "ntrue_seen_min", "ntrue_seen_max",
        "hidden_seen_min", "hidden_seen_max",
        "sched_thin", "opplvl_thin", "promo_thin", "card_thin"]
# AMATEUR PEDIGREE, from Wikidata (see scripts/scraper/scripts/13_wikidata_amateur.py).
# The corpus starts at a man's professional debut, so the first thing anyone
# knows about a prospect — that he boxed for his country — is invisible to every
# other feature here. On the quoted test set, when exactly one of the two has an
# amateur international behind him he wins 73.4% of 458 bouts, and 82.3% of 141
# when he medalled.
#
# Only appearances DATED BEFORE THE BOUT are ever visible: a 2012 fight must not
# know about a 2016 medal. And "no pedigree" is 0, not NaN, on purpose — the
# Olympic rosters Wikidata imports are exhaustive, so the absence of an
# appearance is a fact about the man rather than a hole in our crawl, and a
# score of 0 names no corner.
AMAT = ["d_am", "am_min", "am_max", "am_years", "am_n_min", "am_n_max"]
# the three of the above that are fitted in bt_ratings, not in the replay loop
BTX = ["bt8_min", "bt8_max", "d_bt_z"]

TITLE_RUNG = {"other": 1.0, "regional": 1.0, "national": 2.0,
              "continental": 3.0, "international": 4.0, "world": 5.0}

ALL = BASE + RECORD + SOS2 + GLICKO + AGE + LEVEL
NEW = H2H + DUR + FORM + LEVEL2 + ELO2 + BT + MISS
EVERY = ALL + NEW
# the 2026-08-01 groups, always computable — they need no extra columns
EXTRA = LVLR + LVLQ + UNC + RES + CTX + CMP + THIN + AMAT
# only computable on a corpus snapshot that carries the weigh-in columns
EVERY_W = EVERY + WEIGH
# …and the judges' cards
EVERY_S = EVERY_W + SCORE
# …and the officials who worked the bout
EVERY_O = EVERY_S + OFF
# …and what the saved event pages carried
EVERY_C = EVERY_O + CARD
# …and the levels, the uncertainty terms, the residuals and the context rates.
# AMAT is computed (it is in EXTRA, so it is in the matrix) but deliberately NOT
# in the default set: it was measured and it costs 0.0015 on the premium
# holdout. `--feats everyx+amat` reproduces the negative result.
EVERY_X = EVERY_C + LVLR + LVLQ + UNC + RES + CTX + CMP + THIN


# --------------------------------------------------------------------- Glicko-2
_Q = 173.7178
_TAU = 0.5


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi ** 2))


def _new_vol(delta: float, phi: float, v: float, sigma: float) -> float:
    """Illinois solve for the new volatility (Glickman's step 5)."""
    a = math.log(sigma * sigma)
    d2, p2 = delta * delta, phi * phi

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (d2 - p2 - v - ex)
        den = 2.0 * (p2 + v + ex) ** 2
        return num / den - (x - a) / (_TAU * _TAU)

    A = a
    if d2 > p2 + v:
        B = math.log(d2 - p2 - v)
    else:
        k = 1
        while f(a - k * _TAU) < 0 and k < 100:
            k += 1
        B = a - k * _TAU
    fa, fb = f(A), f(B)
    for _ in range(60):
        if abs(B - A) <= 1e-6:
            break
        C = A + (A - B) * fa / (fb - fa)
        fc = f(C)
        if fc * fb <= 0:
            A, fa = B, fb
        else:
            fa /= 2.0
        B, fb = C, fc
    return math.exp(A / 2.0)


class Glicko2:
    """Per-bout Glicko-2. Boxers fight three times a year on no schedule, so a
    rating period of "one bout" is closer to the truth than a calendar quarter,
    and the RD still inflates between fights via the elapsed-time term."""

    def __init__(self) -> None:
        self.r: dict = defaultdict(lambda: 1500.0)
        self.rd: dict = defaultdict(lambda: 350.0)
        self.sig: dict = defaultdict(lambda: 0.06)
        self.last: dict = {}

    def peek(self, f, dt) -> tuple[float, float]:
        """Rating and RD as of `dt`, inflating RD for the layoff without
        touching state (so a look never changes what a later look sees)."""
        rd = self.rd[f]
        if f in self.last:
            days = max((dt - self.last[f]).days, 0)
            # c chosen so an idle year adds ~50 RD, the usual boxing setting
            rd = min(math.sqrt(rd * rd + (50.0 ** 2) * (days / 365.0)), 350.0)
        return self.r[f], rd

    def update(self, a, b, sa: float, dt) -> None:
        ra, rda = self.peek(a, dt)
        rb, rdb = self.peek(b, dt)
        for x, rx, rdx, y, ry, rdy, s in ((a, ra, rda, b, rb, rdb, sa),
                                          (b, rb, rdb, a, ra, rda, 1.0 - sa)):
            mu, phi = (rx - 1500.0) / _Q, rdx / _Q
            muj, phij = (ry - 1500.0) / _Q, rdy / _Q
            gj = _g(phij)
            e = 1.0 / (1.0 + math.exp(-gj * (mu - muj)))
            e = min(max(e, 1e-9), 1 - 1e-9)
            v = 1.0 / (gj * gj * e * (1 - e))
            delta = v * gj * (s - e)
            sig2 = _new_vol(delta, phi, v, self.sig[x])
            phi_star = math.sqrt(phi * phi + sig2 * sig2)
            phi2 = 1.0 / math.sqrt(1.0 / (phi_star ** 2) + 1.0 / v)
            mu2 = mu + phi2 * phi2 * gj * (s - e)
            self.r[x], self.rd[x], self.sig[x] = mu2 * _Q + 1500.0, phi2 * _Q, sig2
        self.last[a] = self.last[b] = dt


# ------------------------------------------------ whole-history Bradley-Terry
def bt_ratings(df: pd.DataFrame, taus=(2.0, 8.0), step_days: int = 30,
               lam: float = 2.0, iters: int = 40) -> dict[str, np.ndarray]:
    """A rating refitted from scratch on the whole past, at monthly checkpoints.

    Elo and Glicko are ONLINE: each rating is a running estimate that saw the
    opponent's rating as it stood on the night, which for a sporadic three-
    fights-a-year sport means most numbers were set by a version of the
    opponent that no longer exists. Bradley-Terry looks at the whole graph at
    once and lets a late result about the opponent correct an early result
    about the fighter — which is the thing BoxRec's own whole-history rating is
    for.

    Fitted by regularized MM (Zermelo's fixed point with a Gamma prior, so a
    2-0 prospect is shrunk to the mean instead of running off to infinity),
    warm-started from the previous checkpoint, everything vectorized through
    bincount. Two decay constants: tau=2y is "who is good now", tau=8y is "who
    has been good", and they disagree about exactly the fighters worth knowing
    about.

    Strictly point-in-time: the ratings a bout sees were fitted on bouts dated
    before the checkpoint that precedes it, so at worst they are a month stale
    and never a day early.
    """
    fighters = pd.unique(pd.concat([df["a"], df["b"]], ignore_index=True))
    code = {f: i for i, f in enumerate(fighters)}
    n_f = len(fighters)
    ia = df["a"].map(code).to_numpy(np.int64)
    ib = df["b"].map(code).to_numpy(np.int64)
    t = df["dt"].to_numpy("datetime64[D]").astype(np.int64).astype(np.float64)
    sa = np.where(df["is_draw"].to_numpy(), 0.5,
                  (df["winner_id"].astype(str).to_numpy()
                   == df["a"].astype(str).to_numpy()).astype(float))

    out = {k: np.full(len(df), np.nan) for k in BT + BTX}
    edges = np.arange(t[0] + step_days, t[-1] + step_days, step_days, dtype=np.float64)

    for tau in taus:
        key = f"d_bt{int(tau)}"
        s = np.ones(n_f)                       # strengths, warm-started
        half = tau * 365.25
        for c in edges:
            # the fit sees bouts strictly BEFORE the checkpoint; the rows it
            # scores are the ones between this checkpoint and the next, so a
            # bout never contributes to the rating that predicts it
            m = int(np.searchsorted(t, c))
            hi = int(np.searchsorted(t, c + step_days))
            if hi <= m or m == 0:
                continue
            w = np.exp(-(c - t[:m]) / half)
            a_, b_, y_ = ia[:m], ib[:m], sa[:m]
            # wins credited to each side, weighted; draws split
            W = np.bincount(a_, w * y_, n_f) + np.bincount(b_, w * (1 - y_), n_f)
            nw = np.bincount(a_, w, n_f) + np.bincount(b_, w, n_f)
            live = nw > 0
            for _ in range(iters):
                den = w / (s[a_] + s[b_])
                D = np.bincount(a_, den, n_f) + np.bincount(b_, den, n_f)
                s = (W + lam) / (D + lam)
                s /= s[live].mean()
            theta = np.log(np.maximum(s, 1e-12)) * _Q
            theta[~live] = np.nan              # never seen → no opinion
            sl = slice(m, hi)
            out[key][sl] = theta[ia[sl]] - theta[ib[sl]]
            if tau == taus[-1]:
                out["btn_a"][sl] = np.log1p(nw[ia[sl]])
                out["btn_b"][sl] = np.log1p(nw[ib[sl]])
                # the LEVEL of the two whole-history ratings, not only the gap,
                # and the gap divided by how much evidence stands behind it —
                # the same standard error a paired t-test would use
                ta, tb = theta[ia[sl]], theta[ib[sl]]
                out["bt8_min"][sl] = np.minimum(ta, tb)
                out["bt8_max"][sl] = np.maximum(ta, tb)
                se = np.sqrt(1.0 / np.maximum(nw[ia[sl]], 1e-6)
                             + 1.0 / np.maximum(nw[ib[sl]], 1e-6))
                out["d_bt_z"][sl] = (ta - tb) / se
    return out


# ------------------------------------------------------------------ the corpus
def load(tag: str) -> pd.DataFrame:
    """Chronological, and deterministically so. 8.9% of the corpus is dated the
    first of a month (the pre-2010 Wikipedia layer has no day), so thousands of
    bouts tie on the sort key; an unstable sort would give a different replay
    order on a different pandas, and the numbers would drift for no reason."""
    df = pd.read_parquet(STAGING / f"corpus_{tag}.parquet")
    df["dt"] = pd.to_datetime(df["dt"])
    return (df.sort_values(["dt", "a", "b"], kind="stable")
              .reset_index(drop=True))


def symmetrize(df: pd.DataFrame) -> pd.DataFrame:
    """Our rows list the winner first 87% of the time, so a model that always
    answers A scores 0.87. Flip a deterministic half — and flip EVERY paired
    column, not just the ids, or the record-at-time ends up on the wrong man."""
    keys = df["dt"].dt.strftime("%Y%m%d") + "|" + df["a"].astype(str) + "|" + df["b"].astype(str)
    flip = keys.map(lambda k: hashlib.blake2b(k.encode(), digest_size=4).digest()[-1] % 2 == 1).values
    out = df.copy()
    for x, y in PAIRED:
        if x in out.columns and y in out.columns:
            out.loc[flip, [x, y]] = out.loc[flip, [y, x]].values
    return out


def _rate(num, den, default=0.0):
    return (num / den) if den else default


def _mean(d: dict):
    return (d["s"] / d["n"]) if d["n"] else np.nan


def _mm(x, y):
    """(min, max) of the two corners, or (nan, nan) if either is unknown.

    NaN must propagate rather than fall through to the known side: np.fmin
    would quietly return the one value we have, and then "how much do we know
    about this pair" would start naming a corner again — the mistake the age
    group already made once."""
    if x != x or y != y:
        return (np.nan, np.nan)
    return (x, y) if x <= y else (y, x)


def _mm1(x, y):
    """The same, but a single missing side is dropped instead of poisoning both.

    _mm is the right default and it costs something real: when one man is
    making his debut his BoxRec record does not exist, so the PAIR goes NaN and
    the OPPONENT's record — the only thing anyone knows about a showcase debut —
    is thrown away exactly where it is all there is. 20% of the gap to the
    closing line sits on those 6% of bouts.

    Dropping the missing side names no corner: "the record of whichever of the
    two we know" is the same number read from either side of the ring, which is
    precisely the test the age group failed."""
    xa, xb = x == x, y == y
    if xa and xb:
        return (x, y) if x <= y else (y, x)
    if xa:
        return (x, x)
    if xb:
        return (y, y)
    return (np.nan, np.nan)


def dominance(sa: float, a_sc, b_sc, method: str, rf, sched) -> float:
    """How emphatically corner A won, in points per round. 1.0 is a shutout.

    From the judges when there are judges: 19·(A−B)/(A+B) puts 120-108 and
    40-36 both at exactly 1.0, because a shutout is a shutout whatever the
    distance, and it needs no round count of its own — the totals carry it.
    Sanity, measured on the corpus: unanimous decisions average 0.584 a round,
    majority 0.142, split 0.064. That ordering is the whole point; a rating
    that reads one bit per bout cannot see it.

    A stoppage has no full card, so it is scored above a shutout and scaled by
    how early it came: a first-round knockout of a twelve is 1.6, an eleventh-
    round one 1.05. A disqualification says almost nothing about who was
    better, so it is worth a fifth of a round.
    """
    if a_sc == a_sc and b_sc == b_sc and (a_sc + b_sc) > 0:
        # clipped at two points a round: a shutout is one, and the corpus
        # reaches ±7 only where a one-round technical decision leaves a card
        # too small to divide by
        return min(max(19.0 * (a_sc - b_sc) / (a_sc + b_sc), -2.0), 2.0)
    if method in STOP:
        share = 0.0
        if rf == rf and sched == sched and sched > 0:
            share = min(max((rf - 1) / sched, 0.0), 1.0)
        m = 1.0 + 0.6 * (1.0 - share)
        return m if sa >= 1.0 else (-m if sa <= 0.0 else 0.0)
    if method in NO_VERDICT:
        return 0.2 if sa >= 1.0 else (-0.2 if sa <= 0.0 else 0.0)
    # a decision with no card kept: half a point a round, the median decision
    return 0.5 if sa >= 1.0 else (-0.5 if sa <= 0.0 else 0.0)


def amateur_index() -> dict[str, list[tuple]]:
    """fighter id -> [(date, pedigree score), ...] sorted by date.

    Score is 1 for taking part and 2/3/4 for bronze/silver/gold, so a single
    number orders "nobody", "went", and "medalled". Missing file means the
    group is all zeros rather than an error: the replay must still run on a
    machine that has not pulled Wikidata."""
    p = STAGING / "fighter_amateur.parquet"
    if not p.exists():
        return {}
    am = pd.read_parquet(p)
    am = am.sort_values("event_date")
    out: dict[str, list[tuple]] = defaultdict(list)
    for t in am.itertuples(index=False):
        out[str(t.fighter_id)].append((pd.Timestamp(t.event_date),
                                       1.0 + float(t.medal)))
    return dict(out)


def replay(df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0912, PLR0915
    """One chronological pass. Every row is scored on the state BEFORE it."""
    elo = defaultdict(lambda: ELO_INIT)
    elo_mov = defaultdict(lambda: ELO_INIT)
    elo_slow = defaultdict(lambda: ELO_INIT)
    seen, wins, losses = defaultdict(int), defaultdict(int), defaultdict(int)
    ko, koed = defaultdict(int), defaultdict(int)
    sos_sum, sos2_sum = defaultdict(float), defaultdict(float)
    opp_losing, opp_debut = defaultdict(int), defaultdict(int)
    last, last_win, first = {}, {}, {}
    hist = defaultdict(list)
    form = defaultdict(lambda: deque(maxlen=3))
    opp3 = defaultdict(lambda: deque(maxlen=3))
    elo3 = defaultdict(lambda: deque(maxlen=3))
    last_weight: dict = {}
    ctry = defaultdict(lambda: defaultdict(int))
    promo = defaultdict(lambda: defaultdict(int))
    promo_n, city_n, city_home = defaultdict(int), defaultdict(int), defaultdict(int)
    gl = Glicko2()

    # ---- new state
    h2h: dict = defaultdict(lambda: [0, 0.0])        # pair → [meetings, score of the lower id]
    opp: dict = defaultdict(dict)                    # fighter → opponent → [score, n]
    dist_n, rounds_n, mile = defaultdict(int), defaultdict(int), defaultdict(float)
    mile_hist = defaultdict(list)                    # (date, rounds) for the 365d window
    ko_early = defaultdict(int)
    ko_adj = defaultdict(float)                      # stoppage wins, weighted by the chin they broke
    koed_adj = defaultdict(float)                    # stoppage losses, weighted by the power that did it
    streak = defaultdict(int)
    dec_num, dec_den, dec_t = defaultdict(float), defaultdict(float), {}
    last_loss, last_res = {}, {}
    sched_max, sched_sum, sched_n = defaultdict(float), defaultdict(float), defaultdict(int)
    top_opp, beat_top = defaultdict(float), defaultdict(float)
    opp_wr = defaultdict(lambda: {"s": 0.0, "n": 0})
    opp_gl = defaultdict(lambda: {"s": 0.0, "n": 0})
    div_n = defaultdict(lambda: defaultdict(int))
    lbs_hist = defaultdict(lambda: {"s": 0.0, "n": 0})
    # margin ratings: the rating IS the expected margin in points per round, so
    # a difference of 0.4 says "this man should be four rounds up over ten"
    mr = defaultdict(float)
    dom = defaultdict(lambda: {"s": 0.0, "n": 0})
    dom_w = defaultdict(lambda: {"s": 0.0, "n": 0})
    dom_l = defaultdict(lambda: {"s": 0.0, "n": 0})
    dom3 = defaultdict(lambda: deque(maxlen=3))
    MELO_K = 0.10
    # officials. Shrunk toward the running global rate with a pseudo-count, so a
    # referee's third bout does not hand the model a 100% stoppage rate.
    ref = defaultdict(lambda: {"n": 0, "stop": 0, "early": 0, "res": 0.0,
                               "hn": 0, "hw": 0})
    jud = defaultdict(lambda: {"n": 0, "home": 0, "hn": 0, "fav": 0})
    sched_stop = defaultdict(lambda: [0, 0])     # distance → [stoppages, bouts]
    glob = [0, 0]
    PSEUDO = 20.0
    # ---- 2026-08-01 state
    peak = defaultdict(lambda: ELO_INIT)         # highest rating ever reached
    worst_loss: dict = {}                        # the weakest man he lost to
    surp = defaultdict(lambda: deque(maxlen=5))  # result minus what Elo expected
    surp_num, surp_den, surp_t = defaultdict(float), defaultdict(float), {}
    # context → [bouts, upsets, Σ|result − Elo's forecast|]. How well ratings
    # predict on this kind of card, read only off cards of that kind before it.
    ctx = {k: defaultdict(lambda: [0, 0.0, 0.0])
           for k in ("ctry", "promo", "div", "sched")}
    ctx_glob = [0, 0.0, 0.0]
    CTX_PSEUDO = 50.0
    ko_loss_dates = defaultdict(list)            # when he was stopped, and how often
    # the ratings of everyone who has boxed recently, so a rating can be placed
    # against the population that was actually active rather than against 1500
    pop = deque(maxlen=20000)
    pop_stat = [1500.0, 100.0]                   # μ and σ, refreshed periodically
    amat = amateur_index()
    card = df.groupby("event_slug")["a"].transform("size").to_numpy()
    has_w = "a_lbs" in df.columns
    has_s = "a_score" in df.columns
    has_o = "ref_id" in df.columns
    has_c = "a_ctry" in df.columns

    def _shrunk(k, n, prior):
        return (k + PSEUDO * prior) / (n + PSEUDO)

    def _cos(da, db):
        """Do these two men box in the same places? The cosine of their two
        country histograms — 1.0 for two domestic fighters on a domestic card,
        near 0 for two men whose ratings were earned in graphs that barely
        touch, which is exactly when a rating difference means least."""
        if not da or not db:
            return np.nan
        s = sum(da[k] * db[k] for k in (da.keys() & db.keys()))
        if s == 0:
            return 0.0
        na_ = math.sqrt(sum(v * v for v in da.values()))
        nb_ = math.sqrt(sum(v * v for v in db.values()))
        return s / (na_ * nb_)

    def _ctx_stat(kind, key):
        """Upset rate and forecast error on this kind of card so far, shrunk
        toward the running global rate so a country's third bout does not hand
        the model a 0% upset rate."""
        if key is None or key != key:
            return (np.nan, np.nan, np.nan)
        c = ctx[kind][key]
        gu = ctx_glob[1] / ctx_glob[0] if ctx_glob[0] else 0.2
        gm = ctx_glob[2] / ctx_glob[0] if ctx_glob[0] else 0.4
        return ((c[1] + CTX_PSEUDO * gu) / (c[0] + CTX_PSEUDO),
                (c[2] + CTX_PSEUDO * gm) / (c[0] + CTX_PSEUDO),
                math.log1p(c[0]))

    _DEC = 3.0 * 365.25                              # half-life of "form", in days

    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        a, b = r.a, r.b
        ea, eb = elo[a], elo[b]
        na, nb = seen[a], seen[b]
        ga, rda = gl.peek(a, r.dt)
        gb, rdb = gl.peek(b, r.dt)
        ha = _rate(ctry[a][r.country], na) if r.country else 0.0
        hb = _rate(ctry[b][r.country], nb) if r.country else 0.0
        wlbs = DIV_LBS.get(r.div, np.nan)

        # BoxRec's own record on the night. It counts the career before our
        # corpus starts, which for the regional tail is most of it.
        tba = getattr(r, "a_bouts_before", None)
        tbb = getattr(r, "b_bouts_before", None)
        wa, la = getattr(r, "a_wins_before", None), getattr(r, "a_losses_before", None)
        wb, lb = getattr(r, "b_wins_before", None), getattr(r, "b_losses_before", None)
        tba = np.nan if tba is None or pd.isna(tba) else float(tba)
        tbb = np.nan if tbb is None or pd.isna(tbb) else float(tbb)
        wa = np.nan if wa is None or pd.isna(wa) else float(wa)
        la = np.nan if la is None or pd.isna(la) else float(la)
        wb = np.nan if wb is None or pd.isna(wb) else float(wb)
        lb = np.nan if lb is None or pd.isna(lb) else float(lb)

        # age is all-or-nothing on purpose: see the comment on AGE
        age_a = age_b = np.nan
        da, db_ = getattr(r, "a_dob", None), getattr(r, "b_dob", None)
        both_dob = (da is not None and not pd.isna(da)
                    and db_ is not None and not pd.isna(db_))
        if both_dob:
            age_a = (r.dt - pd.Timestamp(da)).days / 365.25
            age_b = (r.dt - pd.Timestamp(db_)).days / 365.25
        hgt_a, hgt_b = getattr(r, "a_height", np.nan), getattr(r, "b_height", np.nan)

        sched = r.sched if r.sched is not None and not pd.isna(r.sched) else np.nan
        wj = np.nan
        if not np.isnan(wlbs):
            pa, pb = last_weight.get(a), last_weight.get(b)
            if pa is not None and pb is not None:
                wj = (wlbs - pa) - (wlbs - pb)

        # ---- head-to-head and common opponents
        pk = (a, b) if str(a) < str(b) else (b, a)
        hn, hs = h2h[pk]
        h2h_score = np.nan
        if hn:
            h2h_score = (hs / hn) if pk[0] == a else (1.0 - hs / hn)
        oa, ob = opp[a], opp[b]
        shared = oa.keys() & ob.keys() if oa and ob else ()
        d_common = np.nan
        if shared:
            d_common = float(np.mean([oa[o][0] / oa[o][1] - ob[o][0] / ob[o][1]
                                      for o in shared]))

        # ---- durability, mileage, form
        cut365 = r.dt - pd.Timedelta(days=365)
        m365a = sum(v for d, v in mile_hist[a] if d > cut365)
        m365b = sum(v for d, v in mile_hist[b] if d > cut365)
        b365a = sum(1 for d in hist[a] if (r.dt - d).days <= 365)
        b365b = sum(1 for d in hist[b] if (r.dt - d).days <= 365)
        b90a = sum(1 for d in hist[a] if (r.dt - d).days <= 90)
        b90b = sum(1 for d in hist[b] if (r.dt - d).days <= 90)

        def wr_dec(f):
            # the decay is applied on the way IN, once per bout, so the ratio
            # already weights the last two years above the first ten; decaying
            # both halves again here would cancel and mean nothing
            return dec_num[f] / dec_den[f] if dec_den[f] > 0 else np.nan

        stepa = (sched - sched_max[a]) if (not np.isnan(sched) and sched_n[a]) else np.nan
        stepb = (sched - sched_max[b]) if (not np.isnan(sched) and sched_n[b]) else np.nan

        # ---- the scales. All six move together or none does: if only one
        # corner were weighed the pattern of which one would name a corner,
        # which is the mistake the age group already made once.
        wgt = ()
        if has_w:
            wa_, wb_ = getattr(r, "a_lbs", np.nan), getattr(r, "b_lbs", np.nan)
            wa_ = np.nan if wa_ is None or pd.isna(wa_) else float(wa_)
            wb_ = np.nan if wb_ is None or pd.isna(wb_) else float(wb_)
            if np.isnan(wa_) or np.isnan(wb_):
                wgt = (np.nan,) * 5 + (0.0,)
            else:
                ha_, hb_ = lbs_hist[a], lbs_hist[b]
                drift = ((wa_ - ha_["s"] / ha_["n"]) - (wb_ - hb_["s"] / hb_["n"])
                         if (ha_["n"] and hb_["n"]) else np.nan)
                oa_ = wa_ - wlbs if not np.isnan(wlbs) else np.nan
                ob_ = wb_ - wlbs if not np.isnan(wlbs) else np.nan
                wgt = (wa_ - wb_, oa_, ob_, oa_ - ob_, drift, 1.0)

        scr = ()
        if has_s:
            scr = (mr[a] - mr[b],
                   _mean(dom[a]) - _mean(dom[b]),
                   _mean(dom_w[a]) - _mean(dom_w[b]),
                   _mean(dom_l[a]) - _mean(dom_l[b]),
                   (np.mean(dom3[a]) if dom3[a] else np.nan)
                   - (np.mean(dom3[b]) if dom3[b] else np.nan),
                   np.log1p(dom[a]["n"]), np.log1p(dom[b]["n"]))

        # Who is actually the home fighter. The flags off the event page settle
        # it outright on 99.9% of bouts; before them this was "whoever has
        # boxed in this country more often", which is a guess about the man's
        # itinerary rather than his passport — and every judge-bias feature was
        # built on top of that guess.
        d_home_true = np.nan
        local = None
        if has_c:
            ac, bc = getattr(r, "a_ctry", None), getattr(r, "b_ctry", None)
            cc = r.country.lower() if isinstance(r.country, str) else ""
            if isinstance(ac, str) and isinstance(bc, str) and cc:
                ah, bh = (ac.lower() == cc), (bc.lower() == cc)
                d_home_true = float(ah) - float(bh)
                if ah != bh:
                    local = a if ah else b
        if local is None and ha != hb:
            local = a if ha > hb else b

        crd = ()
        if has_c:
            bo = getattr(r, "bout_order", np.nan)
            cn = getattr(r, "card_n", np.nan)
            tl = getattr(r, "title_level", None)
            pos = (float(bo) / max(float(cn) - 1.0, 1.0)) if (bo == bo and cn == cn) else np.nan
            crd = (d_home_true,
                   float(d_home_true == d_home_true),
                   TITLE_RUNG.get(tl, 0.0) if isinstance(tl, str) else (
                       0.0 if cn == cn else np.nan),
                   float(isinstance(tl, str)) if cn == cn else np.nan,
                   pos,
                   float(bo == 0) if bo == bo else np.nan)

        off = ()
        if has_o:
            g_stop = glob[0] / glob[1] if glob[1] else 0.35
            rid = getattr(r, "ref_id", None)
            R = ref[rid] if rid else None
            jids = getattr(r, "judge_ids", None)
            panel = [jud[k] for k in jids.split(",")] if isinstance(jids, str) and jids else []
            hh = d_home_true if d_home_true == d_home_true else (
                (ha - hb) if (ha != hb) else np.nan)
            def _panel(key, hkey):
                vals = [_shrunk(p[key], p[hkey], 0.5) for p in panel if p[hkey]]
                return float(np.mean(vals)) if vals else np.nan
            r_home = _shrunk(R["hw"], R["hn"], 0.5) if (R and R["hn"]) else np.nan
            j_home = _panel("home", "hn")
            j_fav = _panel("fav", "n")
            off = (
                _shrunk(R["stop"], R["n"], g_stop) if R and R["n"] else np.nan,
                _shrunk(R["early"], R["n"], g_stop * 0.4) if R and R["n"] else np.nan,
                (R["res"] / R["n"]) if R and R["n"] >= 10 else np.nan,
                np.log1p(R["n"]) if R else np.nan,
                r_home, hh * (r_home - 0.5) if hh == hh else np.nan,
                j_home, hh * (j_home - 0.5) if hh == hh else np.nan,
                j_fav, math.tanh((ea - eb) / 200.0) * (j_fav - 0.5),
                np.log1p(np.mean([p["n"] for p in panel])) if panel else np.nan,
                float(bool(rid) and bool(panel)),
            )

        # ---- 2026-08-01: levels, uncertainty, residuals, context ------------
        # every one of these is either invariant under swapping the corners
        # (min, max, a standard deviation, a rate of the card) or antisymmetric
        # (a difference, or a product of two things that both flip), so none of
        # them can name a corner the way the age group did.
        wr_a_, wr_b_ = _rate(wins[a], na, 0.5), _rate(wins[b], nb, 0.5)
        wrt_a = (wa / tba) if (tba == tba and tba) else np.nan
        wrt_b = (wb / tbb) if (tbb == tbb and tbb) else np.nan
        sos_a_ = _rate(sos_sum[a], na, ELO_INIT)
        sos_b_ = _rate(sos_sum[b], nb, ELO_INIT)
        car_a = (r.dt - first[a]).days if a in first else 0
        car_b = (r.dt - first[b]).days if b in first else 0
        sm_a = sched_max[a] if sched_n[a] else np.nan
        sm_b = sched_max[b] if sched_n[b] else np.nan
        o3a = float(np.mean(opp3[a])) if opp3[a] else np.nan
        o3b = float(np.mean(opp3[b])) if opp3[b] else np.nan
        s5a = float(np.mean(surp[a])) if surp[a] else np.nan
        s5b = float(np.mean(surp[b])) if surp[b] else np.nan
        exp_elo = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))

        elo_mn, elo_mx = _mm(ea, eb)
        gl_mn, gl_mx = _mm(ga, gb)
        rd_mn, rd_mx = _mm(rda, rdb)
        n_mn, n_mx = _mm(float(na), float(nb))
        nt_mn, nt_mx = _mm(tba, tbb)
        car_mn, car_mx = _mm(float(car_a), float(car_b))
        sm_mn, sm_mx = _mm(sm_a, sm_b)
        pk_mn, pk_mx = _mm(peak[a], peak[b])
        wr_mn, wr_mx = _mm(wr_a_, wr_b_)
        wrt_mn, wrt_mx = _mm(wrt_a, wrt_b)
        ko_mn, ko_mx = _mm(_rate(ko[a], na, np.nan), _rate(ko[b], nb, np.nan))
        kd_mn, kd_mx = _mm(_rate(koed[a], na, np.nan), _rate(koed[b], nb, np.nan))
        di_mn, di_mx = _mm(_rate(dist_n[a], na, np.nan), _rate(dist_n[b], nb, np.nan))
        so_mn, so_mx = _mm(sos_a_, sos_b_)
        ow_mn, ow_mx = _mm(_mean(opp_wr[a]), _mean(opp_wr[b]))
        og_mn, og_mx = _mm(_mean(opp_gl[a]), _mean(opp_gl[b]))
        b3_mn, b3_mx = _mm(float(b365a), float(b365b))
        mi_mn, mi_mx = _mm(math.log1p(mile[a]), math.log1p(mile[b]))
        o3_mn, o3_mx = _mm(o3a, o3b)
        s5_mn, s5_mx = _mm(s5a, s5b)

        phi2 = (rda * rda + rdb * rdb) / (_Q * _Q)
        glicko_e = 1.0 / (1.0 + math.exp(-max(min(
            _g(math.sqrt(phi2)) * (ga - gb) / _Q, 30.0), -30.0)))
        rd_norm = math.sqrt(rda * rda + rdb * rdb)
        gaps = [(ea - eb) / 400.0, (ga - gb) / 400.0,
                (elo_mov[a] - elo_mov[b]) / 400.0,
                (elo_slow[a] - elo_slow[b]) / 400.0]
        p_stop_hat = min(max(0.5 * (_rate(ko[a], na, 0.35) + _rate(koed[b], nb, 0.35)
                                    + _rate(ko[b], nb, 0.35) + _rate(koed[a], na, 0.35)),
                             0.0), 1.0)
        cu, cm, cn_ = _ctx_stat("ctry", r.country)
        pu, _, _ = _ctx_stat("promo", r.promoter)
        du, _, _ = _ctx_stat("div", r.div)
        su, _, _ = _ctx_stat("sched", sched)

        # ---- CMP: how comparable are the two ratings, and how recent is the
        # damage. Recomputing μ/σ of the active population from the window
        # rather than carrying running sums: the variance is a difference of
        # two numbers near 2.25e6 and the cancellation would eat it.
        if i % 2000 == 0 and len(pop) >= 100:
            arr = np.fromiter(pop, dtype=float, count=len(pop))
            pop_stat[0] = float(arr.mean())
            pop_stat[1] = float(max(arr.std(), 1e-6))
        ac_, bc_ = getattr(r, "a_ctry", None), getattr(r, "b_ctry", None)
        same_ctry = (float(ac_ == bc_) if isinstance(ac_, str) and isinstance(bc_, str)
                     else np.nan)
        kla = (r.dt - ko_loss_dates[a][-1]).days if ko_loss_dates[a] else 3000
        klb = (r.dt - ko_loss_dates[b][-1]).days if ko_loss_dates[b] else 3000
        kl_mn, kl_mx = _mm(float(sum(1 for d in ko_loss_dates[a] if (r.dt - d).days <= 730)),
                           float(sum(1 for d in ko_loss_dates[b] if (r.dt - d).days <= 730)))
        jm_mn, jm_mx = _mm(b365a * (1.0 - wr_a_), b365b * (1.0 - wr_b_))
        hd_mn, hd_mx = _mm(tba - na if tba == tba else np.nan,
                           tbb - nb if tbb == tbb else np.nan)
        ez_mn, ez_mx = _mm((ea - pop_stat[0]) / pop_stat[1],
                           (eb - pop_stat[0]) / pop_stat[1])
        _tl = getattr(r, "title_level", None)
        t_rung = TITLE_RUNG.get(_tl, 0.0) if isinstance(_tl, str) else np.nan

        # ---- THIN: what is knowable about a man with no record — which is
        # everything about the man opposite him and the card he is on
        wts_mn, wts_mx = _mm1(wrt_a, wrt_b)
        nts_mn, nts_mx = _mm1(tba, tbb)
        hds_mn, hds_mx = _mm1(tba - na if tba == tba else np.nan,
                              tbb - nb if tbb == tbb else np.nan)
        thin = (na == 0) or (nb == 0)
        sched_thin = sched if thin else np.nan
        opplvl_thin = gl_mx if thin else np.nan
        promo_thin = (float(promo_n[r.promoter]) if (thin and r.promoter) else np.nan)
        card_thin = float(card[i]) if thin else np.nan

        # ---- AMAT: only what was already on the record before this bell
        def _ped(f):
            hist = amat.get(f)
            if not hist:
                return 0.0, 0.0, np.nan
            seen = [(d, v) for d, v in hist if d < r.dt]
            if not seen:
                return 0.0, 0.0, np.nan
            return (max(v for _, v in seen), float(len(seen)),
                    (r.dt - seen[-1][0]).days / 365.25)
        am_a, amn_a, amy_a = _ped(a)
        am_b, amn_b, amy_b = _ped(b)
        am_mn, am_mx = (am_a, am_b) if am_a <= am_b else (am_b, am_a)
        amn_mn, amn_mx = (amn_a, amn_b) if amn_a <= amn_b else (amn_b, amn_a)

        rows.append((
            # ---- BASE
            ea - eb, na - nb,
            _rate(wins[a], na, 0.5) - _rate(wins[b], nb, 0.5),
            ((r.dt - last[a]).days if a in last else 400) - ((r.dt - last[b]).days if b in last else 400),
            na, nb, ha - hb,
            (promo[a][r.promoter] - promo[b][r.promoter]) if r.promoter else 0,
            promo_n[r.promoter] if r.promoter else 0,
            (city_home[r.city] / city_n[r.city]) if (r.city and city_n[r.city] >= 20) else np.nan,
            sched,
            _rate(ko[a], na) - _rate(ko[b], nb),
            _rate(koed[a], na) - _rate(koed[b], nb),
            _rate(sos_sum[a], na, ELO_INIT) - _rate(sos_sum[b], nb, ELO_INIT),
            losses[a] - losses[b],
            ((r.dt - first[a]).days if a in first else 0) - ((r.dt - first[b]).days if b in first else 0),
            wlbs,
            (np.mean(form[a]) if form[a] else 0.5) - (np.mean(form[b]) if form[b] else 0.5),
            (np.mean(opp3[a]) if opp3[a] else ELO_INIT) - (np.mean(opp3[b]) if opp3[b] else ELO_INIT),
            ((ea - elo3[a][0]) if elo3[a] else 0.0) - ((eb - elo3[b][0]) if elo3[b] else 0.0),
            np.log1p((r.dt - last[a]).days) if a in last else np.nan,
            np.log1p((r.dt - last[b]).days) if b in last else np.nan,
            ((r.dt - last_win[a]).days if a in last_win else 1500)
            - ((r.dt - last_win[b]).days if b in last_win else 1500),
            sum(1 for d in hist[a] if (r.dt - d).days <= 730)
            - sum(1 for d in hist[b] if (r.dt - d).days <= 730),
            int((a in last and (r.dt - last[a]).days > 180)
                and (b in last and (r.dt - last[b]).days > 180)),
            # ---- RECORD (BoxRec, on the night)
            tba - tbb, tba, tbb, la - lb,
            (wa / tba if tba else np.nan) - (wb / tbb if tbb else np.nan),
            float(la == 0) if not np.isnan(la) else np.nan,
            float(lb == 0) if not np.isnan(lb) else np.nan,
            (tba - na) - (tbb - nb),
            # ---- SOS2
            _rate(sos2_sum[a], na, ELO_INIT) - _rate(sos2_sum[b], nb, ELO_INIT),
            _rate(opp_losing[a], na) - _rate(opp_losing[b], nb),
            _rate(opp_debut[a], na) - _rate(opp_debut[b], nb),
            # ---- GLICKO
            ga - gb, rda, rdb, (ga - 2 * rda) - (gb - 2 * rdb),
            # ---- AGE
            age_a - age_b, age_a, age_b,
            (hgt_a - hgt_b) if (not pd.isna(hgt_a) and not pd.isna(hgt_b)) else np.nan,
            # ---- LEVEL
            float(not np.isnan(sched)), wj,
            # ---- H2H
            hn, h2h_score, len(shared), d_common,
            # ---- DUR
            (float(koed[a] == 0) - float(koed[b] == 0)) if (na and nb) else np.nan,
            _rate(dist_n[a], na, np.nan) - _rate(dist_n[b], nb, np.nan),
            _rate(mile[a], rounds_n[a], np.nan) - _rate(mile[b], rounds_n[b], np.nan),
            np.log1p(mile[a]) - np.log1p(mile[b]),
            m365a - m365b,
            _rate(ko_early[a], na, np.nan) - _rate(ko_early[b], nb, np.nan),
            _rate(ko_adj[a], na, np.nan) - _rate(ko_adj[b], nb, np.nan),
            _rate(koed_adj[a], na, np.nan) - _rate(koed_adj[b], nb, np.nan),
            # ---- FORM
            streak[a] - streak[b],
            (wr_dec(a) if a in dec_t else np.nan) - (wr_dec(b) if b in dec_t else np.nan),
            b365a - b365b, b90a - b90b,
            ((r.dt - last_loss[a]).days if a in last_loss else 3000)
            - ((r.dt - last_loss[b]).days if b in last_loss else 3000),
            last_res.get(a, np.nan) - last_res.get(b, np.nan)
            if (a in last_res and b in last_res) else np.nan,
            # ---- LEVEL2
            card[i],
            (sched_max[a] - sched_max[b]) if (sched_n[a] and sched_n[b]) else np.nan,
            (_rate(sched_sum[a], sched_n[a], np.nan)
             - _rate(sched_sum[b], sched_n[b], np.nan)),
            stepa, stepb,
            (top_opp[a] - top_opp[b]) if (na and nb) else np.nan,
            (beat_top[a] - beat_top[b]) if (wins[a] and wins[b]) else np.nan,
            _mean(opp_wr[a]) - _mean(opp_wr[b]),
            _mean(opp_gl[a]) - _mean(opp_gl[b]),
            div_n[a][r.div] - div_n[b][r.div],
            _rate(div_n[a][r.div], na, np.nan) - _rate(div_n[b][r.div], nb, np.nan),
            # ---- ELO2
            elo_mov[a] - elo_mov[b], elo_slow[a] - elo_slow[b],
            # ---- MISS
            float(both_dob),
            # ---- WEIGH (absent unless the snapshot carries the scales)
            *wgt,
            # ---- SCORE (absent unless the snapshot carries the cards)
            *scr,
            # ---- OFF (absent unless the snapshot carries the officials)
            *off,
            # ---- CARD (absent unless the event pages were re-parsed)
            *crd,
            # ---- LVLR: the level of the two ratings, not only the gap
            elo_mn, elo_mx, gl_mn, gl_mx, rd_mn, rd_mx, n_mn, n_mx,
            nt_mn, nt_mx, car_mn, car_mx, sm_mn, sm_mx, pk_mn, pk_mx,
            # ---- LVLQ: the level of the rates (bt8_min/max come from
            # bt_ratings and are merged after the loop)
            wr_mn, wr_mx, wrt_mn, wrt_mx, ko_mn, ko_mx, kd_mn, kd_mx,
            di_mn, di_mx, so_mn, so_mx, ow_mn, ow_mx, og_mn, og_mx,
            b3_mn, b3_mx, mi_mn, mi_mx, o3_mn, o3_mx,
            # ---- UNC (d_bt_z comes from bt_ratings)
            glicko_e,
            (ga - gb) / rd_norm if rd_norm > 0 else np.nan,
            (ea - eb) / (rda + rdb) * 100.0 if (rda + rdb) > 0 else np.nan,
            (ea - sos_a_) - (eb - sos_b_),
            float(np.std(gaps)),
            (ea - eb) * (elo_mn - ELO_INIT) / 400.0,
            (ga - gb) * (sched / 12.0) if sched == sched else np.nan,
            # ---- RES
            s5a - s5b,
            ((surp_num[a] / surp_den[a]) if surp_den[a] > 0 else np.nan)
            - ((surp_num[b] / surp_den[b]) if surp_den[b] > 0 else np.nan),
            s5_mn, s5_mx,
            peak[a] - peak[b], (peak[a] - ea) - (peak[b] - eb),
            worst_loss.get(a, 2200.0) - worst_loss.get(b, 2200.0),
            # ---- CTX
            cu, cm, cn_, pu, du, su,
            (d_home_true * math.tanh((ea - eb) / 200.0)
             if d_home_true == d_home_true else np.nan),
            p_stop_hat, (ea - eb) * p_stop_hat,
            # ---- CMP
            same_ctry, _cos(ctry[a], ctry[b]), float(kla - klb),
            kl_mn, kl_mx, jm_mn, jm_mx, hd_mn, hd_mx, ez_mn, ez_mx,
            (ea - eb) * cu, (ga - gb) * t_rung,
            # ---- THIN
            wts_mn, wts_mx, nts_mn, nts_mx, hds_mn, hds_mx,
            sched_thin, opplvl_thin, promo_thin, card_thin,
            # ---- AMAT
            am_a - am_b, am_mn, am_mx,
            # how long ago the amateur career was, for whichever man had one:
            # requiring BOTH would have populated it on 8 rows in 40,000, and
            # "the years since the ex-international's last international" is
            # the same number read from either corner
            (np.nanmean([amy_a, amy_b])
             if (amy_a == amy_a or amy_b == amy_b) else np.nan),
            amn_mn, amn_mx,
        ))

        # ------------------------------------------------------------- update
        sa = 0.5 if r.is_draw else (1.0 if str(r.winner_id) == str(a) else 0.0)
        meth = (r.method or "")
        stopped = meth in STOP
        no_verdict = meth in NO_VERDICT
        rf = getattr(r, "round_finished", np.nan)
        rounds = (float(rf) if rf is not None and not pd.isna(rf)
                  else (float(sched) if not np.isnan(sched) else np.nan))

        exp = exp_elo
        elo[a] = ea + ELO_K * (sa - exp)
        elo[b] = eb + ELO_K * ((1 - sa) - (1 - exp))
        peak[a] = max(peak[a], elo[a]); peak[b] = max(peak[b], elo[b])
        # what the ratings expected, against what happened. A 12-0 record built
        # on nobody and a 9-3 record built on contenders look the same in every
        # rate above; this is the number that separates them.
        for x, s, ex in ((a, sa, exp), (b, 1.0 - sa, 1.0 - exp)):
            surp[x].append(s - ex)
            if x in surp_t:
                k = math.exp(-(r.dt - surp_t[x]).days / _DEC)
                surp_num[x] *= k; surp_den[x] *= k
            surp_num[x] += s - ex; surp_den[x] += 1.0; surp_t[x] = r.dt
        if sa <= 0.0:
            worst_loss[a] = min(worst_loss.get(a, 1e9), eb)
        if sa >= 1.0:
            worst_loss[b] = min(worst_loss.get(b, 1e9), ea)
        # how well the ratings did on this kind of card — strictly after the row
        # is written, so a card never contributes to the rate that scores it
        upset = float((sa >= 1.0 and ea < eb) or (sa <= 0.0 and eb < ea))
        mae = abs(sa - exp)
        ctx_glob[0] += 1; ctx_glob[1] += upset; ctx_glob[2] += mae
        for kind, key in (("ctry", r.country), ("promo", r.promoter),
                          ("div", r.div), ("sched", sched)):
            if key is None or key != key:
                continue
            c = ctx[kind][key]
            c[0] += 1; c[1] += upset; c[2] += mae
        # the same update with two other settings: a stoppage is stronger
        # evidence than a split decision, and a slow K remembers longer
        ema, emb = elo_mov[a], elo_mov[b]
        expm = 1.0 / (1.0 + 10 ** ((emb - ema) / 400.0))
        kmul = 1.5 if stopped else (0.7 if no_verdict else 1.0)
        elo_mov[a] = ema + ELO_K * kmul * (sa - expm)
        elo_mov[b] = emb + ELO_K * kmul * ((1 - sa) - (1 - expm))
        esa, esb = elo_slow[a], elo_slow[b]
        exps = 1.0 / (1.0 + 10 ** ((esb - esa) / 400.0))
        elo_slow[a] = esa + 12.0 * (sa - exps)
        elo_slow[b] = esb + 12.0 * ((1 - sa) - (1 - exps))
        gl.update(a, b, sa, r.dt)

        # opposition quality, one level deeper, measured BEFORE this bout
        sos2_sum[a] += _rate(sos_sum[b], nb, ELO_INIT)
        sos2_sum[b] += _rate(sos_sum[a], na, ELO_INIT)
        opp_losing[a] += int(losses[b] > wins[b]); opp_losing[b] += int(losses[a] > wins[a])
        opp_debut[a] += int(nb == 0); opp_debut[b] += int(na == 0)
        opp_wr[a]["s"] += _rate(wins[b], nb, 0.5); opp_wr[a]["n"] += 1
        opp_wr[b]["s"] += _rate(wins[a], na, 0.5); opp_wr[b]["n"] += 1
        opp_gl[a]["s"] += gb; opp_gl[a]["n"] += 1
        opp_gl[b]["s"] += ga; opp_gl[b]["n"] += 1
        if not na:
            top_opp[a] = beat_top[a] = ELO_INIT - 300.0
        if not nb:
            top_opp[b] = beat_top[b] = ELO_INIT - 300.0
        top_opp[a] = max(top_opp[a], eb); top_opp[b] = max(top_opp[b], ea)
        if sa >= 1.0:
            beat_top[a] = max(beat_top[a], eb)
        if sa <= 0.0:
            beat_top[b] = max(beat_top[b], ea)
        sos_sum[a] += eb; sos_sum[b] += ea

        # chin and power, each adjusted by what the other man had shown. A
        # stoppage over someone who had never been stopped is worth more than
        # one over a man who is stopped every time out.
        chin_b = 1.0 - _rate(koed[b], nb, 0.25)
        chin_a = 1.0 - _rate(koed[a], na, 0.25)
        pow_b = _rate(ko[b], nb, 0.35)
        pow_a = _rate(ko[a], na, 0.35)
        if stopped:
            if sa >= 1.0:
                ko[a] += 1; koed[b] += 1
                ko_adj[a] += chin_b; koed_adj[b] += pow_a
                ko_loss_dates[b].append(r.dt)
                if rounds == rounds and rounds <= 3:
                    ko_early[a] += 1
            elif sa <= 0.0:
                ko[b] += 1; koed[a] += 1
                ko_adj[b] += chin_a; koed_adj[a] += pow_b
                ko_loss_dates[a].append(r.dt)
                if rounds == rounds and rounds <= 3:
                    ko_early[b] += 1
        elif not no_verdict:
            dist_n[a] += 1; dist_n[b] += 1
        if rounds == rounds:
            mile[a] += rounds; mile[b] += rounds
            rounds_n[a] += 1; rounds_n[b] += 1
            mile_hist[a].append((r.dt, rounds)); mile_hist[b].append((r.dt, rounds))

        h2h[pk][0] += 1
        h2h[pk][1] += sa if pk[0] == a else (1.0 - sa)
        for x, y, s in ((a, b, sa), (b, a, 1.0 - sa)):
            e = opp[x].get(y)
            if e is None:
                opp[x][y] = [s, 1]
            else:
                e[0] += s; e[1] += 1

        for x, s in ((a, sa), (b, 1.0 - sa)):
            if x in dec_t:
                k = math.exp(-(r.dt - dec_t[x]).days / _DEC)
                dec_num[x] *= k; dec_den[x] *= k
            dec_num[x] += s; dec_den[x] += 1.0; dec_t[x] = r.dt
            last_res[x] = s
            streak[x] = (max(streak[x], 0) + 1 if s >= 1.0 else
                         (min(streak[x], 0) - 1 if s <= 0.0 else 0))
        if sa <= 0.0: last_loss[a] = r.dt
        if sa >= 1.0: last_loss[b] = r.dt

        if not np.isnan(sched):
            sched_max[a] = max(sched_max[a], sched); sched_max[b] = max(sched_max[b], sched)
            sched_sum[a] += sched; sched_sum[b] += sched
            sched_n[a] += 1; sched_n[b] += 1
        div_n[a][r.div] += 1; div_n[b][r.div] += 1
        if has_w:
            if not np.isnan(wa_):
                lbs_hist[a]["s"] += wa_; lbs_hist[a]["n"] += 1
            if not np.isnan(wb_):
                lbs_hist[b]["s"] += wb_; lbs_hist[b]["n"] += 1
        if has_o:
            # strictly after the row is written: an official's tendency is read
            # from the bouts he worked BEFORE this one and never including it
            base = sched_stop[sched if sched == sched else -1]
            exp_stop = (base[0] / base[1]) if base[1] >= 50 else (
                glob[0] / glob[1] if glob[1] else 0.35)
            base[0] += stopped; base[1] += 1
            glob[0] += stopped; glob[1] += 1
            if rid:
                R = ref[rid]
                R["n"] += 1
                R["stop"] += stopped
                R["early"] += int(stopped and rounds == rounds and rounds <= 3)
                R["res"] += stopped - exp_stop
                if local is not None and not r.is_draw:
                    R["hn"] += 1
                    R["hw"] += int(str(r.winner_id) == str(local))
            ja = getattr(r, "judge_a", None)
            jb = getattr(r, "judge_b", None)
            if panel and isinstance(ja, str) and isinstance(jb, str):
                try:
                    sa_l = [float(x) for x in ja.split(",")]
                    sb_l = [float(x) for x in jb.split(",")]
                except ValueError:
                    sa_l = sb_l = []
                # the man the ratings preferred — and when they preferred
                # neither there is no such man, so the judge's "does he go with
                # the favourite" rate must not be updated at all. `>=` made it
                # fall through to corner A and the rate then depended on which
                # way round the bout was written down.
                hi = (a if ea > eb else b) if ea != eb else None
                for p, xa, xb in zip(panel, sa_l, sb_l):
                    if xa == xb:
                        continue
                    picked = a if xa > xb else b
                    if hi is not None:
                        p["n"] += 1
                        p["fav"] += int(picked == hi)
                    if local is not None:
                        p["hn"] += 1
                        p["home"] += int(picked == local)
        if has_s:
            dm = dominance(sa, getattr(r, "a_score", np.nan),
                           getattr(r, "b_score", np.nan), meth, rf, sched)
            err = dm - (mr[a] - mr[b])
            mr[a] += MELO_K * err / 2.0
            mr[b] -= MELO_K * err / 2.0
            for x, v in ((a, dm), (b, -dm)):
                dom[x]["s"] += v; dom[x]["n"] += 1
                dom3[x].append(v)
                tgt = dom_w[x] if v > 0 else (dom_l[x] if v < 0 else None)
                if tgt is not None:
                    tgt["s"] += v; tgt["n"] += 1

        pop.append(ea); pop.append(eb)
        seen[a] += 1; seen[b] += 1
        wins[a] += sa >= 1.0; wins[b] += sa <= 0.0
        if sa <= 0.0: losses[a] += 1
        if sa >= 1.0: losses[b] += 1
        last[a] = last[b] = r.dt
        first.setdefault(a, r.dt); first.setdefault(b, r.dt)
        hist[a].append(r.dt); hist[b].append(r.dt)
        if sa >= 1.0: last_win[a] = r.dt
        if sa <= 0.0: last_win[b] = r.dt
        form[a].append(sa); form[b].append(1 - sa)
        opp3[a].append(eb); opp3[b].append(ea)
        elo3[a].append(ea); elo3[b].append(eb)
        if not np.isnan(wlbs):
            last_weight[a] = last_weight[b] = wlbs
        if r.country:
            ctry[a][r.country] += 1; ctry[b][r.country] += 1
        if r.promoter:
            promo[a][r.promoter] += 1; promo[b][r.promoter] += 1; promo_n[r.promoter] += 1
        if r.city and ha != hb:
            # only when there IS a local man. The tie used to fall through to
            # "corner A", which made the whole feature depend on which way round
            # the bout was written down — mirror_check.py caught it at 0.55 of
            # relative error, and a bout entered the other way round got a
            # different number for a quantity that is a fact about the city.
            city_n[r.city] += 1
            city_home[r.city] += (sa >= 1.0) if ha > hb else (sa <= 0.0)

    full = EVERY
    if has_w:
        full = EVERY_C if has_c else (EVERY_O if has_o else (EVERY_S if has_s else EVERY_W))
    full = full + EXTRA
    merged = set(BT) | set(BTX)
    out = pd.DataFrame(rows, columns=[c for c in full if c not in merged])
    for k, v in bt_ratings(df).items():
        out[k] = v
    return out[full]


def label(df: pd.DataFrame) -> np.ndarray:
    """1 = corner A won. Draws are 0 here; callers that score against a two-way
    market drop them, callers that want three outcomes use is_draw directly."""
    return np.where(df["is_draw"], 0, (df["winner_id"].astype(str) == df["a"].astype(str)).astype(int))
