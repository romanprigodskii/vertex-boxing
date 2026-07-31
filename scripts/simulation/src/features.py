"""One chronological replay, every feature, one place.

There were five copies of replay() with drifting defaults, so two commits'
numbers were never comparable and a fix like "the dates are truncated" had to be
made five times. Experiments are thin wrappers over this now.

Everything here is knowable BEFORE the opening bell. Judges, referee, weigh-in
weights and the finish are deliberately absent: they are on the event page but
not on the card, so a model that uses them is explaining the past, not
predicting the future.

And "knowable before the bell" is not the same as "not contaminated by the
future". Whether OUR database happens to hold a man's date of birth is decided
by a crawl that ran in 2026 and targeted the fighters the odds feed quotes —
so the shape of what is missing is hindsight, and it leaked the winner. See
AGE below.
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
FEATS_VERSION = 3

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
          ("a_lbs", "b_lbs")]

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

ALL = BASE + RECORD + SOS2 + GLICKO + AGE + LEVEL
NEW = H2H + DUR + FORM + LEVEL2 + ELO2 + BT + MISS
EVERY = ALL + NEW
# only computable on a corpus snapshot that carries the weigh-in columns
EVERY_W = EVERY + WEIGH


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

    out = {k: np.full(len(df), np.nan) for k in BT}
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
    card = df.groupby("event_slug")["a"].transform("size").to_numpy()
    has_w = "a_lbs" in df.columns

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
        ))

        # ------------------------------------------------------------- update
        sa = 0.5 if r.is_draw else (1.0 if str(r.winner_id) == str(a) else 0.0)
        meth = (r.method or "")
        stopped = meth in STOP
        no_verdict = meth in NO_VERDICT
        rf = getattr(r, "round_finished", np.nan)
        rounds = (float(rf) if rf is not None and not pd.isna(rf)
                  else (float(sched) if not np.isnan(sched) else np.nan))

        exp = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))
        elo[a] = ea + ELO_K * (sa - exp)
        elo[b] = eb + ELO_K * ((1 - sa) - (1 - exp))
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
                if rounds == rounds and rounds <= 3:
                    ko_early[a] += 1
            elif sa <= 0.0:
                ko[b] += 1; koed[a] += 1
                ko_adj[b] += chin_a; koed_adj[a] += pow_b
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
        if r.city:
            city_n[r.city] += 1
            city_home[r.city] += (sa >= 1.0) if ha >= hb else (sa <= 0.0)

    full = EVERY_W if has_w else EVERY
    out = pd.DataFrame(rows, columns=[c for c in full if c not in BT])
    for k, v in bt_ratings(df).items():
        out[k] = v
    return out[full]


def label(df: pd.DataFrame) -> np.ndarray:
    """1 = corner A won. Draws are 0 here; callers that score against a two-way
    market drop them, callers that want three outcomes use is_draw directly."""
    return np.where(df["is_draw"], 0, (df["winner_id"].astype(str) == df["a"].astype(str)).astype(int))
