#!/usr/bin/env bash
# Every number in docs/REPORT.md, from the frozen `l6` snapshot, into results/.
#
#   ./reproduce.sh              everything, in order (about 2.5 h on an 8-core M3)
#   ./reproduce.sh level_cut    one step by name, assuming the ones it reads exist
#
# The data is not in this repository (see REPORT.md, "Data"), so this runs only
# where imports/staging/ holds corpus_l6.parquet and proboxingodds_v2.parquet.
# The feature matrices are cached per tag and feature version; the first run
# without them replays the corpus, both orientations, which adds about an hour.
#
# The bench is deterministic: LightGBM is pinned to row-wise histograms with
# `deterministic`, the replay sorts every set it sums over, and each bootstrap
# has a fixed seed. The scoreboard published on 2026-08-04 re-ran on 2026-09-21
# identical in every field to 16 significant digits (one log-loss differed in the
# 17th); results/environment.json records the versions these were made with.
#
# The random search of docs/search_protocol.md is not a step here: it is a
# separate experiment with its own protocol, run by scripts/search.py.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
R=results
L=logs
mkdir -p "$R" "$R/checks" "$L"

step() {                       # step <name> <script> [args…]
  local name=$1; shift
  if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$name" ]; then return 0; fi
  echo "== $name  $(date '+%H:%M:%S')"
  "$PY" -u "$@" > "$L/$name.log" 2>&1
}
ONLY=${1:-}

"$PY" - > "$R/environment.json" <<'EOF'
import importlib, json, platform
env = {"python": platform.python_version(), "machine": platform.machine()}
for m in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "pyarrow"):
    env[m] = importlib.import_module(m).__version__
print(json.dumps(env, indent=1))
EOF

# 1. The scoreboard: the model against the closing line, and the blend. Three
#    readings of the board — the close (the worst price of the ten books), the
#    best price on the board, and the open — because they answer different
#    questions and a claim has to say which one it means.
for price in close best open; do
  step "scoreboard_$price" scripts/market_eval.py --tag l6 --tta --mirror --xt --blend \
       --seeds 3 --price "$price" --label "final-$price" --json "$R/scoreboard_$price.json"
done

# 1b. Robustness: the same scoreboard on every price source merged — the
#     ProBoxingOdds re-crawl, OddsPortal and Oddschecker (odds_merge.py). A
#     second feed changes WHICH bouts are priced, so this is a different test
#     set, not a refinement of the first; it is here to show the conclusions do
#     not depend on one site's coverage.
#
#     The merge read −0.002 against −0.021, and feed_check.py shows why: the
#     OddsPortal rows put the favourite on the wrong corner often enough that
#     on bouts only OddsPortal prices the "market" scores worse than a coin. So
#     the merge is kept as the record of that fault, and the robustness check
#     that counts is the ProBoxingOdds re-crawl on its own (+872 bouts to the
#     join, the parser fixed, no second source).
if [ -z "$ONLY" ] || [ "$ONLY" = scoreboard_allodds ]; then
  echo "== scoreboard_allodds  $(date '+%H:%M:%S')"
  VERTEX_ODDS=all "$PY" -u scripts/market_eval.py --tag l6 --tta --mirror --xt --blend \
    --seeds 3 --label final-close-allodds --json "$R/robustness_all_odds.json" \
    > "$L/scoreboard_allodds.log" 2>&1
fi
mkdir -p "$R/diagnostics"
step feed_check scripts/feed_check.py final-close final-close-allodds \
     --json "$R/diagnostics/merged_feed.json"
if [ -z "$ONLY" ] || [ "$ONLY" = scoreboard_pbo3 ]; then
  echo "== scoreboard_pbo3  $(date '+%H:%M:%S')"
  VERTEX_ODDS="$(cd ../.. && pwd)/imports/staging/proboxingodds_v3.parquet" \
    "$PY" -u scripts/market_eval.py --tag l6 --tta --mirror --xt --blend \
    --seeds 3 --label final-close-pbo3 --json "$R/robustness_pbo_v3.json" \
    > "$L/scoreboard_pbo3.log" 2>&1
fi

# 1c. A third price feed, from a different source altogether: BetsAPI, which is
#     Bet365's own line with the price at the moment the bout went in-play as the
#     close (backfill_betsapi.py; prices from late 2020 only). Scored by the SAME
#     model — the cutoff is pinned to the published one — so on the bouts both
#     feeds price only the price differs, and betsapi_check.py says whether the
#     two feeds agree, and what the model does on the bouts only Bet365 prices.
if [ -z "$ONLY" ] || [ "$ONLY" = scoreboard_betsapi ]; then
  echo "== scoreboard_betsapi  $(date '+%H:%M:%S')"
  VERTEX_ODDS="$(cd ../.. && pwd)/imports/staging/odds_external/betsapi.parquet" \
    "$PY" -u scripts/market_eval.py --tag l6 --tta --mirror --xt --blend \
    --seeds 3 --cutoff 2023-06-10 --label final-close-betsapi \
    --json "$R/robustness_betsapi.json" > "$L/scoreboard_betsapi.log" 2>&1
fi
step betsapi_check scripts/betsapi_check.py final-close final-close-betsapi \
     --json "$R/diagnostics/betsapi.json"

# 1c. The margin carried by each of the three readings of the board.
step board_margin scripts/board_margin.py --json "$R/board_margins.json"

# 2. The level cut and the step from closing-line value to money, both on the
#    closing-price predictions written by step 1.
step level_cut scripts/regional.py final-close --tag l6 --json "$R/level_cut.json"
step clv_money scripts/clv_money.py final-close --tag l6 --json "$R/clv_money.json"

# 2b. Is the gap calibration, margin, or information? Three readings of the same
#     saved predictions: the calibration slope with an oracle bound on what any
#     calibrator could recover; where on our own axes the gap sits; and the
#     model, the bookmaker and a zero-margin Polymarket price on the same bouts.
step calibration scripts/calibration.py final-close --json "$R/calibration.json"
step where scripts/where.py final-close --tag l6
step polymarket scripts/polymarket_eval.py final-close --tag l6
mkdir -p "$R/diagnostics"
for c in where polymarket; do
  [ -f "$L/$c.log" ] && cp "$L/$c.log" "$R/diagnostics/$c.txt"
done

# 3. The level rule on an era it was never chosen on: a model refitted to
#    2021-06-10, the rule copied unchanged onto 2021-06 → 2023-06.
step rule_oos scripts/rule_oos.py --json "$R/rule_oos.json"

# 4. The deployment number (yearly walk-forward retraining) and what the
#    post-bell leak is worth when it is put back, both against the same base.
step lab scripts/lab.py --tag l6 --feats everyz --exp deploy,leak-back --seeds 3 \
     --json "$R/deploy_and_leak.json"

# 5. Where the model knows something the price does not: the blend weight
#    fitted separately inside slices that are all known before the bell.
step lam_slice scripts/lam_slice.py --tag l6 --feats everyz --json "$R/lambda_by_slice.json"

# 6. Correctness. mirror_check exits non-zero if any column fails to mirror;
#    leak_check prints its verdicts and the log is kept as the record.
step mirror_check scripts/mirror_check.py l6
step leak_check scripts/leak_check.py l6 --feats everyz
for c in mirror_check leak_check; do
  [ -f "$L/$c.log" ] && cp "$L/$c.log" "$R/checks/$c.txt"
done

# 6b. REPORT section 9: the e-value audits. Three extra models, each trained
#     without the features settled in fight week (weigh-in, referee, the card's
#     judges, running order), because the bets they score are struck at the
#     opening price. The first reads 2023-06 on, the other two the windows of
#     docs/evalue_protocol_2.md. Every scorer validates on simulated outcomes
#     before it reads a real one.
PBO3="$(cd ../.. && pwd)/imports/staging/proboxingodds_v3.parquet"
for m in "openinfo-close 2023-06-10" "win-a-model 2016-06-10" "win-b-model 2021-06-10"; do
  set -- $m
  if [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; then
    echo "== $1  $(date '+%H:%M:%S')"
    VERTEX_ODDS="$PBO3" "$PY" -u scripts/market_eval.py --tag l6 --tta --mirror --xt --seeds 3 \
      --cutoff "$2" --drop weigh+ref+judc+cardpos --label "$1" > "$L/$1.log" 2>&1
  fi
done
step evalue_null scripts/ev_audit.py --null 1000 --json "$R/evalue_null.json"
step evalue_audit scripts/ev_audit.py --json "$R/evalue_audit.json"
step evalue_followup scripts/ev_followup.py --json "$R/evalue_followup.json"
step window_a_null scripts/ev_window.py --label win-a-model --odds proboxingodds_v3.parquet \
     --from 2016-06-10 --to 2020-06-10 --null 1000 --json "$R/window_a_null.json"
step window_b_null scripts/ev_window.py --label win-b-model --odds odds_external/betsapi.parquet \
     --from 2021-06-10 --to 2023-06-10 --null 1000 --json "$R/window_b_null.json"
step window_a scripts/ev_window.py --label win-a-model --odds proboxingodds_v3.parquet \
     --from 2016-06-10 --to 2020-06-10 --real --json "$R/window_a.json"
step window_b scripts/ev_window.py --label win-b-model --odds odds_external/betsapi.parquet \
     --from 2021-06-10 --to 2023-06-10 --real --json "$R/window_b.json"

# 7. The figure and the table of every published number with its source, both
#    written from results/ alone.
step figures scripts/figures.py
step cite_numbers scripts/cite_numbers.py
echo "done  $(date '+%H:%M:%S')"
