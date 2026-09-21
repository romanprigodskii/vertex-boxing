"""The report's figure, drawn from results/ and nothing else.

One chart: closing-line value by the scheduled distance of the fight, a point and
its 95% interval per distance. It is written as plain SVG rather than through a
plotting library so that the file is small, diffable, and renders the same on
GitHub in either theme — a light and a dark copy, which the README picks between
with <picture>.

  python3 scripts/figures.py          # results/level_cut.json → docs/figures/
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
OUT = HERE.parents[1] / "docs" / "figures"

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "muted": "#8a8984", "grid": "#e6e5e1", "zero": "#b4b3ad",
              "series": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "muted": "#8f8e87", "grid": "#2e2e2c", "zero": "#5a5955",
             "series": "#3987e5"},
}
FONT = ("ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Helvetica, Arial, sans-serif")

# the slices regional.py writes, in the order a reader walks up the sport
DISTANCES = [("4-6 раундов (клубный)", "4–6 rounds", "club"),
             ("8 раундов (регионал)", "8 rounds", "regional"),
             ("10 раундов (нац./конт.)", "10 rounds", "national"),
             ("12 раундов (титульный)", "12 rounds", "title")]


def clv_by_distance(theme: str) -> str:
    t = THEMES[theme]
    rows = {r["slice"]: r for r in json.loads((RESULTS / "level_cut.json")
                                              .read_text())["by_distance"]}
    pts = [(rows[k], label, kind) for k, label, kind in DISTANCES]

    W, H = 720, 420
    x0, x1, y0, y1 = 92, 690, 104, 330          # plot box: left, right, top, baseline
    vmax = 0.025
    ticks = [0.0, 0.005, 0.010, 0.015, 0.020, 0.025]

    def y(v: float) -> float:
        return y1 - (v / vmax) * (y1 - y0)

    step = (x1 - x0) / len(pts)
    xs = [x0 + step * (i + 0.5) for i in range(len(pts))]

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" role="img" aria-labelledby="t d" '
         f'font-family="{FONT}">',
         '<title id="t">Closing-line value rises with the level of the fight</title>',
         '<desc id="d">Mean closing-line value per bet, with 95% bootstrap intervals: '
         + "; ".join(f"{label}, {r['bets']} bets, {r['clv']:+.4f} "
                     f"[{r['clv_ci'][0]:+.4f}, {r['clv_ci'][1]:+.4f}]"
                     for r, label, _ in pts) + '.</desc>',
         f'<rect width="{W}" height="{H}" rx="12" fill="{t["surface"]}"/>',
         f'<text x="28" y="42" font-size="17" font-weight="600" fill="{t["ink"]}">'
         'Closing-line value rises with the level of the fight</text>',
         f'<text x="28" y="66" font-size="12.5" fill="{t["ink2"]}">Mean movement of '
         'the price towards the bet between open and close, in probability · '
         '95% bootstrap intervals</text>']

    # recessive grid and a slightly firmer zero line
    for v in ticks:
        yy = y(v)
        o.append(f'<line x1="{x0}" x2="{x1}" y1="{yy:.1f}" y2="{yy:.1f}" '
                 f'stroke="{t["zero"] if v == 0 else t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{x0 - 12}" y="{yy + 4:.1f}" font-size="11.5" '
                 f'text-anchor="end" fill="{t["ink2"]}" '
                 f'font-variant-numeric="tabular-nums">{v:.3f}</text>')

    for (r, label, kind), xx in zip(pts, xs):
        lo, hi, v = r["clv_ci"][0], r["clv_ci"][1], r["clv"]
        # the interval: a 2px line with short caps, then the point with a
        # surface ring so it separates from its own whisker
        o.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y(lo):.1f}" y2="{y(hi):.1f}" '
                 f'stroke="{t["series"]}" stroke-width="2" stroke-linecap="round"/>')
        for c in (lo, hi):
            o.append(f'<line x1="{xx - 7:.1f}" x2="{xx + 7:.1f}" y1="{y(c):.1f}" '
                     f'y2="{y(c):.1f}" stroke="{t["series"]}" stroke-width="2" '
                     f'stroke-linecap="round"/>')
        o.append(f'<circle cx="{xx:.1f}" cy="{y(v):.1f}" r="6" fill="{t["series"]}" '
                 f'stroke="{t["surface"]}" stroke-width="2"/>')
        o.append(f'<text x="{xx + 14:.1f}" y="{y(v) + 4:.1f}" font-size="12.5" '
                 f'font-weight="600" fill="{t["ink"]}" '
                 f'font-variant-numeric="tabular-nums">{v:+.4f}</text>')
        o.append(f'<text x="{xx:.1f}" y="{y1 + 26}" font-size="12.5" font-weight="600" '
                 f'text-anchor="middle" fill="{t["ink"]}">{label}</text>')
        o.append(f'<text x="{xx:.1f}" y="{y1 + 44}" font-size="11.5" '
                 f'text-anchor="middle" fill="{t["ink2"]}">{kind} · {r["bets"]} bets</text>')

    o.append(f'<text x="28" y="{H - 22}" font-size="11" fill="{t["muted"]}">'
             'Priced test bouts, Jun 2023 – Jul 2026. A bet is struck at the open when the '
             "model's probability beats the price by 2 points.</text>")
    o.append("</svg>")
    return "\n".join(o) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        dst = OUT / f"clv_by_distance-{theme}.svg"
        dst.write_text(clv_by_distance(theme))
        print(f"wrote {dst.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
