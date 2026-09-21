"""The model, the bookmaker and a price with no margin in it, on the same bouts.

Everything else in this project scores against a number that has a margin baked
into it, which has to be modelled away before it means anything — and the whole
comparison then rests on which de-vig was chosen. A Polymarket price has nothing
to remove: it is a traded probability on a two-sided book. So this is the one
instrument that can separate "the model trails the market" from "the model
trails a particular reading of the bookmaker's number".

It is also tiny — see run_polymarket.py for the sizing — so read the intervals,
not the point estimates. Ten to twenty-five bouts cannot resolve 0.002. What
they CAN do is answer a qualitative question that nothing else here answers, and
incidentally audit the de-vig: if the de-vigged bookmaker close and the
zero-margin price agree on these fights, the de-vig is doing its job.

  python3 scripts/polymarket_eval.py final-close --tag l6
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(ROOT / "scripts" / "simulation" / "scripts"))
import market_eval as ME  # noqa: E402

CACHE = ROOT / "imports" / "staging"
LAST_N = 25          # fills averaged into "the price before the bell"


def ll(p, y):
    return -np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-9, 1))


def boot(d, n=4000, seed=42):
    rng = np.random.default_rng(seed)
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return tuple(np.percentile(b, [2.5, 97.5]))


def main() -> None:
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "l6"
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a != tag]
    label = args[0] if args else "final-close"
    pm = pd.read_parquet(CACHE / "polymarket_boxing.parquet")
    # the rows the predictions were keyed on: key_corp indexes this frame
    df = pd.read_parquet(CACHE / f"sym_{tag}.parquet",
                         columns=["dt", "a", "b", "a_name", "b_name",
                                  "winner_id", "is_draw"])
    df["dt"] = pd.to_datetime(df["dt"])
    d = np.load(CACHE / "preds" / f"{label}.npz", allow_pickle=True)
    p_by_row = dict(zip(d["key_corp"].tolist(), d["p_corp"].tolist()))

    df["na"] = df["a_name"].map(ME.norm)
    df["nb"] = df["b_name"].map(ME.norm)

    rows, misses = [], []
    for (q, a_nm, b_nm, slug), g in pm.groupby(
            ["question", "a_name", "b_name", "event_slug"], sort=False):
        na, nb = ME.norm(a_nm), ME.norm(b_nm)
        if not na or not nb:
            continue
        end = pd.to_datetime(g["end_date"].iloc[0], utc=True)
        if end is None or pd.isna(end):
            # a market with no end date cannot be placed against a bout date
            misses.append(("no end date", q[:52]))
            continue
        # candidate bouts within three days, either corner order
        w = df[(df["dt"] >= end.tz_localize(None) - pd.Timedelta(days=3))
               & (df["dt"] <= end.tz_localize(None) + pd.Timedelta(days=3))]
        hit = None
        for t in w.itertuples(index=True):
            if t.na is None or t.nb is None:
                continue
            if (ME._same_man(na, t.na) and ME._same_man(nb, t.nb)):
                hit, flip = t, False
                break
            if (ME._same_man(na, t.nb) and ME._same_man(nb, t.na)):
                hit, flip = t, True
                break
        if hit is None:
            misses.append((str(end.date()), q[:52]))
            continue
        if hit.is_draw or hit.Index not in p_by_row:
            continue
        # strictly before 00:00 UTC on the day of the bout — see run_polymarket.py
        cut = pd.Timestamp(hit.dt.date(), tz="UTC")
        pre = g[g["dt"] < cut].sort_values("dt")
        if len(pre) < 10:
            continue
        p_last = float(pre["p_a"].tail(LAST_N).median())
        # The TRUE opener, which no odds feed records: the first fills this
        # market ever took. Our `open_*` column is whatever proboxingodds
        # happened to write down, and nothing says that was early.
        p_open = float(pre["p_a"].head(LAST_N).median())
        # p_a is the price of the market's FIRST name; orient it onto corner A
        y = int(str(hit.winner_id) == str(hit.a))
        rows.append({"slug": slug, "q": q, "dt": hit.dt.date(), "row": hit.Index,
                     "n_pre": len(pre),
                     "p_poly": 1.0 - p_last if flip else p_last,
                     "p_poly_open": 1.0 - p_open if flip else p_open,
                     "first_ts": pd.Timestamp(pre["dt"].iloc[0]),
                     "p_model": p_by_row[hit.Index], "y": y,
                     "two_sided": bool(g["two_sided"].iloc[0]),
                     "volume": float(g["volume"].iloc[0])})

    if not rows:
        print("no Polymarket market matched a corpus bout")
        return
    r = pd.DataFrame(rows).drop_duplicates("row", keep="first")
    y = r["y"].to_numpy()
    print(f"matched {len(r)} bouts · {len(misses)} markets found no corpus bout\n")
    print(f"{'дата':11s} {'бой':40s} {'Poly':>6s} {'модель':>7s} {'A выиграл':>10s}")
    for t in r.sort_values("dt").itertuples(index=False):
        print(f"{str(t.dt):11s} {t.q[:40]:40s} {t.p_poly:6.3f} {t.p_model:7.3f} "
              f"{'да' if t.y else 'нет':>10s}")

    lp, lm = ll(r["p_poly"].to_numpy(), y), ll(r["p_model"].to_numpy(), y)
    print(f"\n{'':22s} {'log-loss':>9s} {'Brier':>8s}")
    print(f"{'Polymarket (без маржи)':22s} {lp.mean():9.4f} "
          f"{((r['p_poly'] - y) ** 2).mean():8.4f}")
    print(f"{'наша модель':22s} {lm.mean():9.4f} "
          f"{((r['p_model'] - y) ** 2).mean():8.4f}")
    dd = lm - lp
    lo, hi = boot(dd)
    print(f"  модель − Polymarket: {dd.mean():+.4f} [95% {lo:+.4f}, {hi:+.4f}]"
          + ("  (Polymarket лучше)" if lo > 0 else
             "  (неотличимо — что при таком n и ожидается)"))

    # the same bouts against the bookmaker, if the odds feed also priced them
    j = ME.join_odds(df, verbose=False)
    same = ~j["swap"].to_numpy(bool)
    ca = np.where(same, j["close_a"], j["close_b"]).astype(float)
    cb = np.where(same, j["close_b"], j["close_a"]).astype(float)
    ok = np.isfinite(ca) & np.isfinite(cb) & (ca > 1) & (cb > 1)
    oa = np.where(same, j["open_a"], j["open_b"]).astype(float)
    ob = np.where(same, j["open_b"], j["open_a"]).astype(float)
    ook = ok & np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1)
    book = pd.DataFrame({"row": j["index"].to_numpy()[ok],
                         "p_book": ME.devig(1 / ca[ok], 1 / cb[ok], "power")})
    opens = pd.DataFrame({"row": j["index"].to_numpy()[ook],
                          "p_feed_open": ME.devig(1 / oa[ook], 1 / ob[ook], "power")})
    m = r.merge(book, on="row", how="inner")
    if len(m) >= 5:
        ym = m["y"].to_numpy()
        print(f"\n--- на {len(m)} боях, где есть ОБЕ цены ---")
        for nm, col in (("Polymarket", "p_poly"), ("букмекер (power de-vig)", "p_book"),
                        ("наша модель", "p_model")):
            print(f"  {nm:24s} log-loss {ll(m[col].to_numpy(), ym).mean():.4f}")
        gap = np.abs(m["p_poly"].to_numpy() - m["p_book"].to_numpy())
        print(f"  |Polymarket − де-вигнутый букмекер|: медиана {np.median(gap):.3f}, "
              f"среднее {gap.mean():.3f}")
        print("  ^ маленькое расхождение означает, что де-виг работает")
    # --- how early is our "open", really? -----------------------------------
    o = r.merge(opens, on="row", how="inner").merge(book, on="row", how="inner")
    if len(o) >= 5:
        yo = o["y"].to_numpy()
        print(f"\n--- НАСТОЯЩЕЕ открытие против того, что записал фид "
              f"({len(o)} боёв) ---")
        print(f"{'дата':11s} {'первые сделки':>14s} {'open фида':>10s} "
              f"{'close фида':>11s} {'A выиграл':>10s}")
        for t in o.sort_values("dt").itertuples(index=False):
            print(f"{str(t.dt):11s} {t.p_poly_open:14.3f} {t.p_feed_open:10.3f} "
                  f"{t.p_book:11.3f} {'да' if t.y else 'нет':>10s}")
        for nm, col in (("Polymarket, первые сделки", "p_poly_open"),
                        ("Polymarket, перед гонгом", "p_poly"),
                        ("open фида", "p_feed_open"),
                        ("close фида", "p_book")):
            print(f"  {nm:26s} log-loss {ll(o[col].to_numpy(), yo).mean():.4f}")
        d_of = np.abs(o["p_poly_open"] - o["p_feed_open"])
        d_cf = np.abs(o["p_poly_open"] - o["p_book"])
        print(f"\n  |первые сделки − open фида| : медиана {np.median(d_of):.3f}")
        print(f"  |первые сделки − close фида|: медиана {np.median(d_cf):.3f}")
        print("  ^ если первое ЗАМЕТНО меньше второго, фид пишет уже "
              "устоявшуюся цену,\n    и настоящий опенер мягче того, "
              "по чему мы считаем эдж")

    if misses:
        print(f"\nне сматчились ({len(misses)}):")
        for dt_, q in misses[:12]:
            print(f"  {dt_}  {q}")


if __name__ == "__main__":
    main()
