"""The registry: every hypothesis the e-value audit will test, fixed before it runs.

Committed before a single Bet365 closing price in the confirmatory window was
scored against an outcome. docs/evalue_protocol.md says what was and was not seen
before this file was written, how each hypothesis becomes an e-value, and what
counts as a rejection. This file is the part of that protocol a machine reads:
the audit (ev_audit.py) takes its hypotheses from here and from nowhere else, so
the commit that holds this file is the timestamp.

Two kinds of hypothesis, and they ask different things.

  M  "the model knows something the price does not, in this slice". The
     alternative is the blend of the model and the price, σ(λ·logit p + (1-λ)·
     logit q), mixed uniformly over λ in LAMBDAS. Two-sided in the outcome by
     construction; one-sided in that it only ever leans towards the model.

  B  "the price is biased against a side named before the bell", with no model
     in it at all: the favourite, the home fighter, the fighter on a long win
     streak. The alternative moves that side's probability by δ logits, mixed
     uniformly over δ in DELTAS; `sign` says which way the mechanism predicts,
     and 0 means both ways were plausible, so the mixture covers both and pays
     for it.

Every filter and every side is a function of what was known before the bell:
the corpus record at the time, the card, the venue, and the Bet365 prices up to
the close. Nothing here reads a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "imports" / "staging"

LAMBDAS = np.round(np.arange(0.05, 0.501, 0.05), 2)   # ten blend weights
DELTAS = np.round(np.arange(0.05, 0.501, 0.05), 2)    # ten logit shifts
BETAS = np.array([0.80, 0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.20])  # control
ALPHA = 0.05

CONFIRM = (pd.Timestamp("2023-06-10"), pd.Timestamp("2025-12-31"))  # (after, through)
SEEN_FROM = pd.Timestamp("2026-01-01")      # continuation: seen in aggregate
ASIA = {"JP", "TH", "PH", "CN", "KR", "ID", "IN", "KZ", "UZ", "MN", "VN", "MY", "SG", "HK", "TW"}
STOP = {"ko", "tko", "rtd"}


@dataclass(frozen=True)
class H:
    hid: str
    kind: str                                # "M" or "B"
    name: str
    mask: Callable[[pd.DataFrame], pd.Series]
    why: str
    side: Callable[[pd.DataFrame], pd.Series] | None = None   # B only: 1 = corner A
    sign: int = 0                            # B only: +1 underpriced, -1 over, 0 either


# ------------------------------------------------------------------ covariates
def _history(sym: pd.DataFrame) -> pd.DataFrame:
    """Each corner's form going into the bout, from the corpus alone.

    Processed a DAY at a time: every bout on a date reads the state from before
    that date, so a fighter's second bout of a one-night tournament cannot see
    the first. A no-contest leaves a streak as it was; a draw ends it."""
    st: dict = {}
    out = {k: np.zeros(len(sym)) for k in
           ("a_streak", "b_streak", "a_last_l", "b_last_l", "a_last_stop_l",
            "b_last_stop_l", "a_kow", "b_kow", "a_n", "b_n")}
    order = np.argsort(sym["dt"].to_numpy(), kind="stable")
    dts = sym["dt"].to_numpy()[order]
    A, B = sym["a"].to_numpy()[order], sym["b"].to_numpy()[order]
    W = sym["winner_id"].astype(str).to_numpy()[order]
    D = sym["is_draw"].to_numpy(bool)[order]
    MTH = sym["method"].astype(str).str.lower().to_numpy()[order]
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and dts[j] == dts[i]:
            j += 1
        for k in range(i, j):
            r = order[k]
            for c, f in (("a", A[k]), ("b", B[k])):
                s = st.get(f, (0, 0, 0, 0, 0))
                out[f"{c}_streak"][r], out[f"{c}_last_l"][r] = s[0], s[1]
                out[f"{c}_last_stop_l"][r], out[f"{c}_kow"][r], out[f"{c}_n"][r] = s[2], s[3], s[4]
        for k in range(i, j):
            for f, o in ((A[k], B[k]), (B[k], A[k])):
                s = st.get(f, (0, 0, 0, 0, 0))
                if D[k]:
                    s = (0, 0, 0, s[3], s[4] + 1)
                elif W[k] == str(f):
                    s = (s[0] + 1, 0, 0, s[3] + (MTH[k] in STOP), s[4] + 1)
                elif W[k] == str(o):
                    s = (0, 1, int(MTH[k] in STOP), s[3], s[4] + 1)
                st[f] = s                    # no contest (no winner): unchanged
        i = j
    return pd.DataFrame(out, index=sym.index)


def covariates(tag: str = "l6") -> pd.DataFrame:
    from src import features as F  # noqa: PLC0415
    sym = pd.read_parquet(CACHE / f"sym_{tag}.parquet",
                          columns=["dt", "a", "b", "winner_id", "is_draw", "method",
                                   "a_bouts_before", "b_bouts_before",
                                   "a_losses_before", "b_losses_before",
                                   "a_ctry", "b_ctry", "country"])
    f = pd.read_parquet(CACHE / F.cache_name("feats", tag),
                        columns=["sched_rounds", "title_lvl", "is_main", "lay_a", "lay_b",
                                 "weight_lbs", "h2h_n"])
    c = pd.concat([sym[["dt"]], _history(sym)], axis=1)
    c["sched"] = f["sched_rounds"].to_numpy()
    c["belt"] = np.nan_to_num(f["title_lvl"].to_numpy(float), nan=0)
    c["main"] = f["is_main"].to_numpy() == 1
    c["a_bouts"], c["b_bouts"] = sym["a_bouts_before"].to_numpy(), sym["b_bouts_before"].to_numpy()
    c["a_unb"] = (sym["a_losses_before"].fillna(1) == 0).to_numpy()
    c["b_unb"] = (sym["b_losses_before"].fillna(1) == 0).to_numpy()
    c["a_lay"], c["b_lay"] = np.expm1(f["lay_a"].to_numpy()), np.expm1(f["lay_b"].to_numpy())
    c["venue"] = sym["country"].astype(str).str.upper().to_numpy()
    a_c = sym["a_ctry"].astype(str).str.upper().to_numpy()
    b_c = sym["b_ctry"].astype(str).str.upper().to_numpy()
    c["a_home"], c["b_home"] = a_c == c["venue"].to_numpy(), b_c == c["venue"].to_numpy()
    c["a_ctry"], c["b_ctry"] = a_c, b_c
    c["same_ctry"] = a_c == b_c
    c["lbs"] = f["weight_lbs"].to_numpy()
    c["h2h"] = f["h2h_n"].to_numpy()
    c["a_kor"] = c["a_kow"] / c["a_n"].clip(lower=1)
    c["b_kor"] = c["b_kow"] / c["b_n"].clip(lower=1)
    return c


# The audit adds, per bout, from the Bet365 feed: q (power-de-vigged close, P(A)),
# q_open (the same at the open, NaN if none), p (the published model, P(A)).

def _thin(c):
    return np.fmin(c["a_bouts"], c["b_bouts"])


def _fav(c):                   # 1 where A is the closing favourite
    return (c["q"] >= 0.5).astype(int)


def _qfav(c):
    return np.maximum(c["q"], 1 - c["q"])


def _one(ca, cb):
    """Exactly one corner has the property: that corner (1 = A); else NaN."""
    return pd.Series(np.where(ca & ~cb, 1.0, np.where(cb & ~ca, 0.0, np.nan)), index=ca.index)


# ------------------------------------------------------------------ primary
# Tested one by one against 1/(α/4): four questions, a union bound, e ≥ 80.
PRIMARY = ["P1", "P2", "P3", "P4"]
PRIMARY_WHY = {
    "P1": "M on every confirmatory bout. The report's blend result (+0.0043 against "
          "ProBoxingOdds' close), asked of a second bookmaker's close.",
    "P2": "M on the confirmatory bouts ProBoxingOdds does not price at all — the only "
          "bouts whose outcome has never been scored against any price in this project.",
    "P3": "Money at the OPEN: Kelly stakes at Bet365's real opening decimals, the "
          "alternative the blend of the model with the opening price. REPORT 4.3 said "
          "the margin at the open exceeds the movement the model catches; this is the "
          "test that settles it with outcomes.",
    "P4": "Money at the CLOSE: the same at Bet365's real closing decimals.",
}
CONTROL_WHY = ("C1, reported beside P1 and in no family: the price recalibrated with no "
               "model in it, σ(β·logit q), β mixed over BETAS. If this is as large as P1, "
               "P1 measured the de-vig, not the model.")

# ------------------------------------------------------------------ the family
# One family, e-BH at α = 0.05, K = len(FAMILY).
FAMILY: list[H] = [
    # --- level: the axis the thesis was about (REPORT 4.2)
    H("M01", "M", "scheduled 6 rounds or fewer, or unknown",
      lambda c: ~(c["sched"] > 6), "club level: the thesis said the price is softest here; the "
      "report found λ = 0 on this slice"),
    H("M02", "M", "scheduled 8 rounds", lambda c: c["sched"] == 8, "regional level"),
    H("M03", "M", "scheduled 10 rounds", lambda c: c["sched"] == 10, "national and continental level"),
    H("M04", "M", "scheduled 12 rounds", lambda c: c["sched"] == 12,
      "title level: where the report found the closing-line value largest"),
    H("M05", "M", "no belt", lambda c: c["belt"] == 0, "the bulk of the priced set"),
    H("M06", "M", "regional, other or national belt", lambda c: c["belt"].between(1, 2),
      "lower belts: the report's λ was 0.39 here on its fitting window"),
    H("M07", "M", "continental, international or world belt", lambda c: c["belt"] >= 3,
      "the most-bet bouts; the most efficient price is expected here"),
    H("M08", "M", "world title", lambda c: c["belt"] == 5, "the sharpest market in the sport"),
    H("M09", "M", "main event", lambda c: c["main"], "the bout the card is sold on"),
    H("M10", "M", "not the main event", lambda c: ~c["main"], "undercard: less money, less attention"),
    H("M11", "M", "upper tier: 12 rounds or a belt at rung 3+",
      lambda c: (c["sched"] >= 12) | (c["belt"] >= 3), "the report's own rule, copied verbatim"),
    H("M12", "M", "lower tier: 8 rounds or fewer, no belt",
      lambda c: ~(c["sched"] > 8) & (c["belt"] == 0), "the report's own rule, copied verbatim"),
    # --- experience
    H("M13", "M", "thinner record under 3 bouts", lambda c: _thin(c) < 3,
      "a rating on two bouts is noise; the market knows who is being managed"),
    H("M14", "M", "thinner record 3-7 bouts", lambda c: _thin(c).between(3, 7), "prospect level"),
    H("M15", "M", "thinner record 8-14 bouts", lambda c: _thin(c).between(8, 14), "established records"),
    H("M16", "M", "thinner record 15+ bouts", lambda c: _thin(c) >= 15,
      "both records long enough for a rating to mean something"),
    H("M17", "M", "both fighters 25+ bouts", lambda c: np.fmin(c["a_bouts"], c["b_bouts"]) >= 25,
      "veterans on both sides"),
    H("M18", "M", "one record at least three times the other",
      lambda c: np.fmax(c["a_bouts"], c["b_bouts"]) >= 3 * np.fmax(np.fmin(c["a_bouts"], c["b_bouts"]), 1),
      "experience mismatch: the matchmaker's intent is in the pairing"),
    # --- form
    H("M19", "M", "either fighter on a win streak of 10+", lambda c: np.fmax(c["a_streak"], c["b_streak"]) >= 10,
      "long streaks are built by matchmaking; the record overstates the fighter"),
    H("M20", "M", "both fighters on win streaks of 5+", lambda c: np.fmin(c["a_streak"], c["b_streak"]) >= 5,
      "two protected records meeting: the matchmaking signal cancels"),
    H("M21", "M", "both unbeaten", lambda c: c["a_unb"] & c["b_unb"], "prospect against prospect"),
    H("M22", "M", "either fighter lost his last bout", lambda c: (c["a_last_l"] == 1) | (c["b_last_l"] == 1),
      "a fresh loss is salient to bettors"),
    H("M23", "M", "either fighter was stopped in his last bout",
      lambda c: (c["a_last_stop_l"] == 1) | (c["b_last_stop_l"] == 1), "chin damage is hard to price"),
    H("M24", "M", "either fighter out for 365+ days", lambda c: np.fmax(c["a_lay"], c["b_lay"]) >= 365,
      "ring rust: a record says nothing about the last year"),
    H("M25", "M", "both fought within 180 days", lambda c: np.fmax(c["a_lay"], c["b_lay"]) < 180,
      "active fighters: the records are current"),
    # --- geography
    H("M26", "M", "same country", lambda c: c["same_ctry"], "the report's λ was 0.01 here"),
    H("M27", "M", "different countries", lambda c: ~c["same_ctry"],
      "cross-border: the report's λ was 0.24 here; one side's record is foreign to the local market"),
    H("M28", "M", "exactly one fighter at home", lambda c: c["a_home"] != c["b_home"],
      "home and away: local money on one side"),
    H("M29", "M", "venue in the United States", lambda c: c["venue"] == "US", "the largest regulated market"),
    H("M30", "M", "venue in Great Britain", lambda c: c["venue"] == "GB", "Bet365's home market"),
    H("M31", "M", "venue in Mexico", lambda c: c["venue"] == "MX", "the largest volume of club boxing"),
    H("M32", "M", "venue in Asia", lambda c: c["venue"].isin(ASIA),
      "thin coverage in English-language sources, which is where the market's knowledge comes from"),
    H("M33", "M", "venue anywhere else", lambda c: ~c["venue"].isin(ASIA | {"US", "GB", "MX"}),
      "the remainder, so the geography partitions"),
    # --- weight
    H("M34", "M", "200 lb and over", lambda c: c["lbs"] >= 200, "heavyweight variance: one punch"),
    H("M35", "M", "126 lb and under", lambda c: c["lbs"] <= 126, "the lighter divisions go the distance more"),
    # --- the price itself (all known at the close)
    H("M36", "M", "closing favourite at 85%+", lambda c: _qfav(c) >= 0.85, "mismatches"),
    H("M37", "M", "closing favourite at 65-85%", lambda c: _qfav(c).between(0.65, 0.85, inclusive="left"),
      "clear favourites"),
    H("M38", "M", "closing favourite under 65%", lambda c: _qfav(c) < 0.65, "competitive bouts"),
    H("M39", "M", "line moved 5+ points from open to close", lambda c: (c["q"] - c["q_open"]).abs() >= 0.05,
      "informed money has already moved it"),
    H("M40", "M", "line moved under 2 points", lambda c: (c["q"] - c["q_open"]).abs() < 0.02,
      "the market learned nothing new, or nobody bet"),
    H("M41", "M", "model and close 10+ points apart", lambda c: (c["p"] - c["q"]).abs() >= 0.10,
      "where the two disagree is where the blend moves most"),
    H("M42", "M", "rematch", lambda c: c["h2h"] >= 1, "the first bout is information both sides have"),
    # --- time, so a result cannot be one stretch of the calendar
    H("M43", "M", "2023-06-11 to 2024-06-30",
      lambda c: c["dt"] <= pd.Timestamp("2024-06-30"), "first year after the cut-off"),
    H("M44", "M", "2024-07-01 to 2025-06-30",
      lambda c: (c["dt"] > pd.Timestamp("2024-06-30")) & (c["dt"] <= pd.Timestamp("2025-06-30")),
      "second year"),
    H("M45", "M", "2025-07-01 to 2025-12-31", lambda c: c["dt"] > pd.Timestamp("2025-06-30"),
      "the last half-year, the model at its stalest"),

    # --- price biases, no model (B). side = the corner the claim is about.
    H("B01", "B", "the closing favourite is underpriced", lambda c: pd.Series(True, index=c.index),
      "favourite-longshot bias: bettors overpay for longshots", side=_fav, sign=+1),
    H("B02", "B", "a favourite at 80%+ is underpriced", lambda c: _qfav(c) >= 0.80,
      "the same bias, strongest in mismatches", side=_fav, sign=+1),
    H("B03", "B", "a fighter on a 10+ win streak, against one who is not, is overpriced",
      lambda c: _one(c["a_streak"] >= 10, c["b_streak"] >= 10).notna(),
      "narrative premium: a long streak sells and was built by matchmaking",
      side=lambda c: _one(c["a_streak"] >= 10, c["b_streak"] >= 10), sign=-1),
    H("B04", "B", "an unbeaten fighter with 10+ bouts, against a beaten one, is overpriced",
      lambda c: _one(c["a_unb"] & (c["a_bouts"] >= 10), c["b_unb"] & (c["b_bouts"] >= 10)).notna(),
      "the zero is marketed; the price pays for the marketing",
      side=lambda c: _one(c["a_unb"] & (c["a_bouts"] >= 10), c["b_unb"] & (c["b_bouts"] >= 10)), sign=-1),
    H("B05", "B", "the home fighter is mispriced (either way)", lambda c: c["a_home"] != c["b_home"],
      "local money overprices him, or home judging is underpriced; both have a mechanism",
      side=lambda c: _one(c["a_home"], c["b_home"]), sign=0),
    H("B06", "B", "the British fighter at home in Britain is overpriced",
      lambda c: (c["venue"] == "GB") & (c["a_home"] != c["b_home"]),
      "Bet365's home market: the most partisan money it takes",
      side=lambda c: _one(c["a_home"], c["b_home"]), sign=-1),
    H("B07", "B", "a fighter stopped in his last bout is overpriced",
      lambda c: _one(c["a_last_stop_l"] == 1, c["b_last_stop_l"] == 1).notna(),
      "damage the record does not show",
      side=lambda c: _one(c["a_last_stop_l"] == 1, c["b_last_stop_l"] == 1), sign=-1),
    H("B08", "B", "a fighter back from 365+ days out is overpriced",
      lambda c: _one(c["a_lay"] >= 365, c["b_lay"] >= 365).notna(),
      "ring rust", side=lambda c: _one(c["a_lay"] >= 365, c["b_lay"] >= 365), sign=-1),
    H("B09", "B", "the fighter with three times the experience is mispriced (either way)",
      lambda c: _one(c["a_bouts"] >= 3 * np.fmax(c["b_bouts"], 1),
                     c["b_bouts"] >= 3 * np.fmax(c["a_bouts"], 1)).notna(),
      "experience is visible to everyone; which way the market errs is not obvious",
      side=lambda c: _one(c["a_bouts"] >= 3 * np.fmax(c["b_bouts"], 1),
                          c["b_bouts"] >= 3 * np.fmax(c["a_bouts"], 1)), sign=0),
    H("B10", "B", "the side the line moved towards (3+ points) is mispriced (either way)",
      lambda c: (c["q"] - c["q_open"]).abs() >= 0.03,
      "the move overshoots, or it stops short", side=lambda c: (c["q"] > c["q_open"]).astype(int), sign=0),
    H("B11", "B", "the world-title favourite is mispriced (either way)", lambda c: c["belt"] == 5,
      "the most-bet market; public money on the name", side=_fav, sign=0),
    H("B12", "B", "the heavyweight underdog is underpriced", lambda c: c["lbs"] >= 200,
      "the puncher's chance", side=lambda c: 1 - _fav(c), sign=+1),
    H("B13", "B", "the underdog on a 5+ win streak is underpriced",
      lambda c: pd.Series(np.where(c["q"] >= 0.5, c["b_streak"], c["a_streak"]) >= 5, index=c.index),
      "form the favourite's name hides", side=lambda c: 1 - _fav(c), sign=+1),
    H("B14", "B", "the fighter with the higher knockout rate is overpriced",
      lambda c: ((c["a_kor"] - c["b_kor"]).abs() >= 0.25) & (np.fmin(c["a_n"], c["b_n"]) >= 5),
      "power is what bettors see", side=lambda c: (c["a_kor"] > c["b_kor"]).astype(int), sign=-1),
    H("B15", "B", "the favourite in a cross-border bout is mispriced (either way)",
      lambda c: ~c["same_ctry"], "one record is foreign to half the market", side=_fav, sign=0),
    H("B16", "B", "the closing favourite on a 6-rounder or shorter is underpriced",
      lambda c: ~(c["sched"] > 6), "club mismatches, where the B-side is booked to lose",
      side=_fav, sign=+1),
]

assert len({h.hid for h in FAMILY}) == len(FAMILY)
K = len(FAMILY)
