"""Every number the report and the README quote, and the file and field it comes from.

A document that cites a number should be able to say where the number lives.
This reads results/*.json and writes results/NUMBERS.md: one row per published
figure, its value as printed, and the path to it. If a result file changes, this
is re-run and the documents are checked against the table — not the other way
round.

  python3 scripts/cite_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"

# regional.py and lam_slice.py label their slices in the language of the lab
# notebook; the published table uses these
EN = {"4-6 раундов (клубный)": "4–6 rounds (club)",
      "8 раундов (регионал)": "8 rounds (regional)",
      "10 раундов (нац./конт.)": "10 rounds (national/continental)",
      "12 раундов (титульный)": "12 rounds (title)",
      "дистанция неизвестна": "distance unknown",
      "менее 8 боёв у слабейшего": "thinner record under 8 bouts",
      "8-15": "thinner record 8–15 bouts", "15-25": "thinner record 15–25 bouts",
      "25+": "thinner record 25+ bouts",
      "без пояса, карточка <8": "no belt, card under 8 bouts",
      "без пояса, карточка 8+": "no belt, card of 8+ bouts",
      "регион./нац. пояс": "regional or national belt",
      "конт./межд./мировой": "continental, international or world belt"}


def en(name: str) -> str:
    return EN.get(name, name)


def load(name: str) -> dict | None:
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def f4(v: float) -> str:
    return f"{v:+.4f}".replace("-", "−")


def ci4(c: list) -> str:
    return f"[{f4(c[0])}, {f4(c[1])}]"


def pct(v: float, d: int = 1) -> str:
    return f"{v * 100:+.{d}f}%".replace("-", "−")


def cipct(c: list, d: int = 1) -> str:
    return f"[{pct(c[0], d)}, {pct(c[1], d)}]"


def main() -> None:  # noqa: PLR0915
    rows: list[tuple[str, str, str]] = []

    def add(name, value, src):
        rows.append((name, value, src))

    for price in ("close", "best", "open"):
        s = load(f"scoreboard_{price}.json")
        if not s:
            continue
        f = f"scoreboard_{price}.json"
        if price == "close":
            add("priced test bouts", f"{s['n_test']:,}", f"{f} · n_test")
            add("corpus holdout bouts", f"{s['n_corpus_test']:,}", f"{f} · n_corpus_test")
            add("premium holdout bouts", f"{s['n_prem']:,}", f"{f} · n_prem")
            add("log-loss, corpus holdout", f"{s['ll_corpus']:.4f}", f"{f} · ll_corpus")
            add("log-loss, premium holdout", f"{s['ll_prem']:.4f}", f"{f} · ll_prem")
            add("log-loss, model on priced bouts", f"{s['ll_model']:.4f}", f"{f} · ll_model")
            add("de-vig chosen on train, its calibration slope",
                f"{s['devig']} {s['devig_slope']:.3f}", f"{f} · devig, devig_slope")
        add(f"log-loss, market ({price})", f"{s['ll_market']:.4f}", f"{f} · ll_market")
        add(f"model − market ({price})", f"{f4(s['gap'])} {ci4(s['ci'])}", f"{f} · gap, ci")
        add(f"blend − market ({price})",
            f"{f4(s['blend_gap'])} {ci4(s['blend_ci'])} at λ {s['blend_w_model']:.2f}",
            f"{f} · blend_gap, blend_ci, blend_w_model")

    bm = load("board_margins.json")
    if bm:
        for price, r in bm.items():
            add(f"margin in the {price} reading of the board", f"{r['margin_mean']:.1%}",
                f"board_margins.json · {price}.margin_mean")

    lc = load("level_cut.json")
    if lc:
        for sec in ("by_distance", "by_record_depth", "by_belt_and_card"):
            for r in lc[sec]:
                add(f"CLV, {en(r['slice'])} ({r['bets']} bets)",
                    f"{f4(r['clv'])} {ci4(r['clv_ci'])} · ROI {pct(r['roi'])}",
                    f"level_cut.json · {sec}[slice={r['slice']}]")
        for r in lc["strategy"]:
            add(f"strategy '{r['filter']}' ({r['bets']} bets)",
                f"CLV {f4(r['clv'])} {ci4(r['clv_ci'])} · ROI {pct(r['roi'])} {cipct(r['roi_ci'])}",
                f"level_cut.json · strategy[filter={r['filter']}]")

    cm = load("clv_money.json")
    if cm:
        for dv, blk in cm["by_method"].items():
            a, u = blk["upper_arithmetic"], blk["upper_tier"]
            add(f"upper tier, the book keeps at the open ({dv})", f4(a["book_keeps"]),
                f"clv_money.json · by_method.{dv}.upper_arithmetic.book_keeps")
            add(f"upper tier, CLV ({dv} at both ends)", f"{f4(a['clv'])} {ci4(a['clv_ci'])}",
                f"clv_money.json · by_method.{dv}.upper_arithmetic.clv, clv_ci")
            add(f"upper tier, net per bet ({dv})", f4(a["net"]),
                f"clv_money.json · by_method.{dv}.upper_arithmetic.net")
            add(f"upper tier, return at the closing price ({dv})",
                f"{pct(u['roi_at_close'])} {cipct(u['roi_at_close_ci'])}",
                f"clv_money.json · by_method.{dv}.upper_tier.roi_at_close")
        u = cm["by_method"]["power"]["upper_tier"]
        add("upper tier, realised return", f"{pct(u['roi'])} {cipct(u['roi_ci'])}",
            "clv_money.json · by_method.*.upper_tier.roi")

    ro = load("rule_oos.json")
    if ro:
        for k in ("all", "low", "top"):
            r = ro["slices"].get(k)
            if r:
                prop = (f" · proportional {pct(r['roi_at_close_proportional'])} "
                        f"{cipct(r['roi_at_close_proportional_ci'])}"
                        if "roi_at_close_proportional" in r else "")
                add(f"out-of-sample {k} ({r['bets']} bets)",
                    f"CLV {f4(r['clv'])} {ci4(r['clv_ci'])} · ROI {pct(r['roi'])} "
                    f"{cipct(r['roi_ci'])} · at close: power {pct(r['roi_at_close'])} "
                    f"{cipct(r['roi_at_close_ci'])}{prop}",
                    f"rule_oos.json · slices.{k}")
        if "clv_ratio_top_over_low" in ro:
            add("out-of-sample CLV, top over bottom", f"×{ro['clv_ratio_top_over_low']:.2f}",
                "rule_oos.json · clv_ratio_top_over_low")

    for f, name in (("robustness_pbo_v3.json", "ProBoxingOdds re-crawl"),
                    ("robustness_all_odds.json", "merged feed (OddsPortal fault, not used)")):
        rb = load(f)
        if rb:
            add(f"robustness, {name}: priced test bouts", f"{rb['n_test']:,}", f"{f} · n_test")
            add(f"robustness, {name}: model − market", f"{f4(rb['gap'])} {ci4(rb['ci'])}",
                f"{f} · gap, ci")
            add(f"robustness, {name}: blend − market",
                f"{f4(rb['blend_gap'])} {ci4(rb['blend_ci'])} at λ {rb['blend_w_model']:.2f}",
                f"{f} · blend_gap, blend_ci")
    fc = load("diagnostics/merged_feed.json")
    if fc:
        for src, r in fc["added_by_source"].items():
            add(f"merged feed, bouts only it prices, source {src} ({r['n']})",
                f"market {r['market']:.3f} · model {r['model']:.3f}",
                f"diagnostics/merged_feed.json · added_by_source.{src}")

    lab = load("deploy_and_leak.json")
    if lab:
        v = lab["variants"]
        if "deploy" in v:
            d = v["deploy"]
            add("retrained yearly: model − market", f"{f4(d['gap_to_close']['gap'])} "
                f"{ci4(d['gap_to_close']['ci'])}", "deploy_and_leak.json · variants.deploy.gap_to_close")
            add("retrained yearly: corpus / premium / priced log-loss",
                f"{d['ll_corpus']:.4f} / {d['ll_prem']:.4f} / {d['ll_quoted']:.4f}",
                "deploy_and_leak.json · variants.deploy.ll_*")
        if "leak-back" in v:
            d = v["leak-back"]
            for k in ("confirm", "select", "prem", "quoted"):
                x = d.get(f"delta_{k}")
                if x:
                    add(f"the leak put back, worth on {k}", f"{f4(x['delta'])} {ci4(x['ci'])}",
                        f"deploy_and_leak.json · variants.leak-back.delta_{k}")

    ls = load("lambda_by_slice.json")
    if ls:
        add("blend weight λ, all priced bouts", f"{ls['global_lambda']:.2f}",
            "lambda_by_slice.json · global_lambda")
        for r in ls["slices"]:
            add(f"λ, {r['slice']} (fit {r['n_fit']} / test {r['n_test']})", f"{r['lambda']:.2f}",
                f"lambda_by_slice.json · slices[slice={r['slice']}]")

    se = load("search.json")
    if se:
        import statistics as st
        screen = [json.loads(x) for x in (R / "search_screen.jsonl").read_text().splitlines()]
        cands = [x for x in screen if x["id"].startswith("c")]
        band = se["seed_band_A_prem"]
        add("search: candidates screened", str(se["n_screened"]), "search.json · n_screened")
        add("search: final configuration under five seeds, window A premium",
            f"{band['min']:.4f}–{band['max']:.4f}", "search.json · seed_band_A_prem")
        add("search: median candidate, window A premium",
            f"{st.median(x['ll_A_prem'] for x in cands):.4f}", "search_screen.jsonl · ll_A_prem")
        add("search: candidates better than the best seed",
            str(sum(x["ll_A_prem"] < band["min"] for x in cands)), "search_screen.jsonl · ll_A_prem")
        for c in se["candidates"]:
            add(f"search: {c['id']}, lead on A / on unseen premium [99%] / survives",
                f"{f4(c['lead_A_prem'])} / {f4(c['delta_B_prem']['delta'])} "
                f"{ci4(c['delta_B_prem']['ci99'])} / {'yes' if c['survives'] else 'no'}",
                f"search.json · candidates[id={c['id']}]")

    cal = load("calibration.json")
    if cal:
        for k in ("corpus", "premium", "quoted"):
            add(f"calibration slope, {k}", f"{cal[k]['calibration_slope']:.3f}",
                f"calibration.json · {k}.calibration_slope")
        add("oracle logit factor on priced bouts buys", f4(cal["quoted"]["oracle_gain"]),
            "calibration.json · quoted.oracle_gain")

    lines = ["# Every published number and where it comes from", "",
             "Written by `scripts/cite_numbers.py` from the JSON files in this folder. "
             "Text diagnostics (`diagnostics/where.txt`, `diagnostics/polymarket.txt`) "
             "and the correctness logs in `checks/` are cited by name in the report.", "",
             "| number | value | source |", "|---|---|---|"]
    lines += [f"| {n} | {v} | `{s}` |" for n, v, s in rows]
    (R / "NUMBERS.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} numbers → results/NUMBERS.md")


if __name__ == "__main__":
    main()
