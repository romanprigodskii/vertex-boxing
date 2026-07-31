"""One chronological replay, every feature, one place.

There were five copies of replay() with drifting defaults, so two commits'
numbers were never comparable and a fix like "the dates are truncated" had to be
made five times. Experiments are thin wrappers over this now.

Everything here is knowable BEFORE the opening bell. Judges, referee, weigh-in
weights and the finish are deliberately absent: they are on the event page but
not on the card, so a model that uses them is explaining the past, not
predicting the future.
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

ELO_K, ELO_INIT = 32.0, 1500.0
# division → its nominal pound limit; a real ordering beats a category code
DIV_LBS = {"minimumweight": 105, "light_flyweight": 108, "flyweight": 112,
           "super_flyweight": 115, "bantamweight": 118, "super_bantamweight": 122,
           "featherweight": 126, "super_featherweight": 130, "lightweight": 135,
           "super_lightweight": 140, "welterweight": 147, "super_welterweight": 154,
           "middleweight": 160, "super_middleweight": 168, "light_heavyweight": 175,
           "cruiserweight": 200, "heavyweight": 250}
STOP = {"ko", "tko", "rtd"}

# a_*/b_* columns that must move together when the corners are swapped
PAIRED = [("a", "b"), ("a_name", "b_name"),
          ("a_bouts_before", "b_bouts_before"),
          ("a_wins_before", "b_wins_before"),
          ("a_losses_before", "b_losses_before"),
          ("a_draws_before", "b_draws_before"),
          ("a_dob", "b_dob"), ("a_height", "b_height"), ("a_stance", "b_stance")]

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
AGE = ["d_age", "age_a", "age_b", "d_height", "stance_clash"]
LEVEL = ["sched_known", "d_weight_jump"]

ALL = BASE + RECORD + SOS2 + GLICKO + AGE + LEVEL


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


# ------------------------------------------------------------------ the corpus
def load(tag: str) -> pd.DataFrame:
    df = pd.read_parquet(STAGING / f"corpus_{tag}.parquet")
    df["dt"] = pd.to_datetime(df["dt"])
    return df.sort_values(["dt"]).reset_index(drop=True)


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


def replay(df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0915
    """One chronological pass. Every row is scored on the state BEFORE it."""
    elo = defaultdict(lambda: ELO_INIT)
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

    rows = []
    for r in df.itertuples(index=False):
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

        age_a = age_b = np.nan
        da, db_ = getattr(r, "a_dob", None), getattr(r, "b_dob", None)
        if da is not None and not pd.isna(da):
            age_a = (r.dt - pd.Timestamp(da)).days / 365.25
        if db_ is not None and not pd.isna(db_):
            age_b = (r.dt - pd.Timestamp(db_)).days / 365.25
        hgt_a, hgt_b = getattr(r, "a_height", np.nan), getattr(r, "b_height", np.nan)
        st_a, st_b = getattr(r, "a_stance", None), getattr(r, "b_stance", None)
        clash = np.nan
        if st_a and st_b and st_a != "unknown" and st_b != "unknown":
            clash = float(st_a != st_b)

        sched = r.sched if r.sched is not None and not pd.isna(r.sched) else np.nan
        wj = np.nan
        if not np.isnan(wlbs):
            pa, pb = last_weight.get(a), last_weight.get(b)
            if pa is not None and pb is not None:
                wj = (wlbs - pa) - (wlbs - pb)

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
            clash,
            # ---- LEVEL
            float(not np.isnan(sched)), wj,
        ))

        # ------------------------------------------------------------- update
        sa = 0.5 if r.is_draw else (1.0 if str(r.winner_id) == str(a) else 0.0)
        exp = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))
        elo[a] = ea + ELO_K * (sa - exp)
        elo[b] = eb + ELO_K * ((1 - sa) - (1 - exp))
        gl.update(a, b, sa, r.dt)
        # opposition quality, one level deeper, measured BEFORE this bout
        sos2_sum[a] += _rate(sos_sum[b], nb, ELO_INIT)
        sos2_sum[b] += _rate(sos_sum[a], na, ELO_INIT)
        opp_losing[a] += int(losses[b] > wins[b]); opp_losing[b] += int(losses[a] > wins[a])
        opp_debut[a] += int(nb == 0); opp_debut[b] += int(na == 0)
        sos_sum[a] += eb; sos_sum[b] += ea
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
        if (r.method or "") in STOP:
            if sa >= 1.0:
                ko[a] += 1; koed[b] += 1
            elif sa <= 0.0:
                ko[b] += 1; koed[a] += 1
        if not np.isnan(wlbs):
            last_weight[a] = last_weight[b] = wlbs
        if r.country:
            ctry[a][r.country] += 1; ctry[b][r.country] += 1
        if r.promoter:
            promo[a][r.promoter] += 1; promo[b][r.promoter] += 1; promo_n[r.promoter] += 1
        if r.city:
            city_n[r.city] += 1
            city_home[r.city] += (sa >= 1.0) if ha >= hb else (sa <= 0.0)

    return pd.DataFrame(rows, columns=ALL)


def label(df: pd.DataFrame) -> np.ndarray:
    """1 = corner A won. Draws are 0 here; callers that score against a two-way
    market drop them, callers that want three outcomes use is_draw directly."""
    return np.where(df["is_draw"], 0, (df["winner_id"].astype(str) == df["a"].astype(str)).astype(int))
