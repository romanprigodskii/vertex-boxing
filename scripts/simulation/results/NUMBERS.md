# Every published number and where it comes from

Written by `scripts/cite_numbers.py` from the JSON files in this folder. Text diagnostics (`diagnostics/where.txt`, `diagnostics/polymarket.txt`) and the correctness logs in `checks/` are cited by name in the report.

| number | value | source |
|---|---|---|
| priced test bouts | 3,288 | `scoreboard_close.json · n_test` |
| corpus holdout bouts | 89,087 | `scoreboard_close.json · n_corpus_test` |
| premium holdout bouts | 12,536 | `scoreboard_close.json · n_prem` |
| log-loss, corpus holdout | 0.3342 | `scoreboard_close.json · ll_corpus` |
| log-loss, premium holdout | 0.2820 | `scoreboard_close.json · ll_prem` |
| log-loss, model on priced bouts | 0.3680 | `scoreboard_close.json · ll_model` |
| de-vig chosen on train, its calibration slope | power 1.021 | `scoreboard_close.json · devig, devig_slope` |
| log-loss, market (close) | 0.3469 | `scoreboard_close.json · ll_market` |
| model − market (close) | −0.0211 [−0.0322, −0.0100] | `scoreboard_close.json · gap, ci` |
| blend − market (close) | +0.0043 [+0.0024, +0.0062] at λ 0.17 | `scoreboard_close.json · blend_gap, blend_ci, blend_w_model` |
| log-loss, market (best) | 0.3478 | `scoreboard_best.json · ll_market` |
| model − market (best) | −0.0202 [−0.0313, −0.0091] | `scoreboard_best.json · gap, ci` |
| blend − market (best) | +0.0047 [+0.0027, +0.0067] at λ 0.18 | `scoreboard_best.json · blend_gap, blend_ci, blend_w_model` |
| log-loss, market (open) | 0.3591 | `scoreboard_open.json · ll_market` |
| model − market (open) | −0.0089 [−0.0206, +0.0026] | `scoreboard_open.json · gap, ci` |
| blend − market (open) | +0.0107 [+0.0063, +0.0150] at λ 0.37 | `scoreboard_open.json · blend_gap, blend_ci, blend_w_model` |
| margin in the close reading of the board | 7.3% | `board_margins.json · close.margin_mean` |
| margin in the best reading of the board | 3.0% | `board_margins.json · best.margin_mean` |
| margin in the open reading of the board | 5.6% | `board_margins.json · open.margin_mean` |
| CLV, 4–6 rounds (club) (283 bets) | +0.0044 [+0.0017, +0.0073] · ROI −14.9% | `level_cut.json · by_distance[slice=4-6 раундов (клубный)]` |
| CLV, 8 rounds (regional) (337 bets) | +0.0075 [+0.0033, +0.0119] · ROI +5.4% | `level_cut.json · by_distance[slice=8 раундов (регионал)]` |
| CLV, 10 rounds (national/continental) (781 bets) | +0.0093 [+0.0060, +0.0127] · ROI +4.1% | `level_cut.json · by_distance[slice=10 раундов (нац./конт.)]` |
| CLV, 12 rounds (title) (468 bets) | +0.0180 [+0.0130, +0.0232] · ROI +13.1% | `level_cut.json · by_distance[slice=12 раундов (титульный)]` |
| CLV, distance unknown (173 bets) | +0.0111 [+0.0042, +0.0182] · ROI −2.6% | `level_cut.json · by_distance[slice=дистанция неизвестна]` |
| CLV, thinner record under 8 bouts (455 bets) | +0.0082 [+0.0049, +0.0115] · ROI −8.7% | `level_cut.json · by_record_depth[slice=менее 8 боёв у слабейшего]` |
| CLV, thinner record 8–15 bouts (800 bets) | +0.0117 [+0.0085, +0.0148] · ROI +3.9% | `level_cut.json · by_record_depth[slice=8-15]` |
| CLV, thinner record 15–25 bouts (652 bets) | +0.0117 [+0.0078, +0.0157] · ROI +8.6% | `level_cut.json · by_record_depth[slice=15-25]` |
| CLV, thinner record 25+ bouts (141 bets) | +0.0059 [−0.0028, +0.0150] · ROI +18.3% | `level_cut.json · by_record_depth[slice=25+]` |
| CLV, no belt, card under 8 bouts (401 bets) | +0.0082 [+0.0040, +0.0128] · ROI +9.3% | `level_cut.json · by_belt_and_card[slice=без пояса, карточка <8]` |
| CLV, no belt, card of 8+ bouts (703 bets) | +0.0067 [+0.0039, +0.0094] · ROI −6.0% | `level_cut.json · by_belt_and_card[slice=без пояса, карточка 8+]` |
| CLV, regional or national belt (269 bets) | +0.0121 [+0.0057, +0.0184] · ROI +1.7% | `level_cut.json · by_belt_and_card[slice=регион./нац. пояс]` |
| CLV, continental, international or world belt (675 bets) | +0.0152 [+0.0112, +0.0194] · ROI +10.9% | `level_cut.json · by_belt_and_card[slice=конт./межд./мировой]` |
| strategy 'all' (2048 bets) | CLV +0.0105 [+0.0084, +0.0125] · ROI +3.6% [−2.2%, +9.5%] | `level_cut.json · strategy[filter=all]` |
| strategy 'ten_plus_or_belt' (1384 bets) | CLV +0.0124 [+0.0097, +0.0153] · ROI +5.6% [−0.9%, +12.4%] | `level_cut.json · strategy[filter=ten_plus_or_belt]` |
| strategy 'upper' (803 bets) | CLV +0.0143 [+0.0104, +0.0182] · ROI +8.7% [−0.0%, +18.0%] | `level_cut.json · strategy[filter=upper]` |
| upper tier, the book keeps at the open (proportional) | +0.0313 | `clv_money.json · by_method.proportional.upper_arithmetic.book_keeps` |
| upper tier, CLV (proportional at both ends) | +0.0143 [+0.0104, +0.0182] | `clv_money.json · by_method.proportional.upper_arithmetic.clv, clv_ci` |
| upper tier, net per bet (proportional) | −0.0169 | `clv_money.json · by_method.proportional.upper_arithmetic.net` |
| upper tier, return at the closing price (proportional) | −0.6% [−1.9%, +0.9%] | `clv_money.json · by_method.proportional.upper_tier.roi_at_close` |
| upper tier, the book keeps at the open (power) | +0.0275 | `clv_money.json · by_method.power.upper_arithmetic.book_keeps` |
| upper tier, CLV (power at both ends) | +0.0180 [+0.0134, +0.0225] | `clv_money.json · by_method.power.upper_arithmetic.clv, clv_ci` |
| upper tier, net per bet (power) | −0.0095 | `clv_money.json · by_method.power.upper_arithmetic.net` |
| upper tier, return at the closing price (power) | −6.1% [−7.7%, −4.4%] | `clv_money.json · by_method.power.upper_tier.roi_at_close` |
| upper tier, realised return | +8.7% [−0.0%, +18.0%] | `clv_money.json · by_method.*.upper_tier.roi` |
| out-of-sample all (1145 bets) | CLV +0.0220 [+0.0179, +0.0264] · ROI +4.0% [−3.5%, +12.0%] · at close: power −6.3% [−8.0%, −4.6%] · proportional +1.0% [−0.4%, +2.5%] | `rule_oos.json · slices.all` |
| out-of-sample low (365 bets) | CLV +0.0136 [+0.0080, +0.0191] · ROI −8.3% [−20.6%, +4.7%] · at close: power −11.8% [−14.6%, −9.0%] · proportional −0.6% [−2.6%, +1.7%] | `rule_oos.json · slices.low` |
| out-of-sample top (461 bets) | CLV +0.0276 [+0.0200, +0.0347] · ROI +9.5% [−2.6%, +22.8%] · at close: power −3.2% [−5.9%, −0.4%] · proportional +2.8% [+0.3%, +5.4%] | `rule_oos.json · slices.top` |
| out-of-sample CLV, top over bottom | ×2.04 | `rule_oos.json · clv_ratio_top_over_low` |
| robustness, ProBoxingOdds re-crawl: priced test bouts | 3,644 | `robustness_pbo_v3.json · n_test` |
| robustness, ProBoxingOdds re-crawl: model − market | −0.0206 [−0.0315, −0.0101] | `robustness_pbo_v3.json · gap, ci` |
| robustness, ProBoxingOdds re-crawl: blend − market | +0.0037 [+0.0022, +0.0052] at λ 0.14 | `robustness_pbo_v3.json · blend_gap, blend_ci` |
| robustness, merged feed (OddsPortal fault, not used): priced test bouts | 3,705 | `robustness_all_odds.json · n_test` |
| robustness, merged feed (OddsPortal fault, not used): model − market | −0.0021 [−0.0151, +0.0107] | `robustness_all_odds.json · gap, ci` |
| robustness, merged feed (OddsPortal fault, not used): blend − market | +0.0163 [+0.0106, +0.0220] at λ 0.36 | `robustness_all_odds.json · blend_gap, blend_ci` |
| merged feed, bouts only it prices, source oddsportal (37) | market 1.022 · model 0.326 | `diagnostics/merged_feed.json · added_by_source.oddsportal` |
| merged feed, bouts only it prices, source oddsportal+proboxingodds (38) | market 0.672 · model 0.290 | `diagnostics/merged_feed.json · added_by_source.oddsportal+proboxingodds` |
| merged feed, bouts only it prices, source proboxingodds (342) | market 0.185 · model 0.197 | `diagnostics/merged_feed.json · added_by_source.proboxingodds` |
| retrained yearly: model − market | −0.0198 [−0.0311, −0.0086] | `deploy_and_leak.json · variants.deploy.gap_to_close` |
| retrained yearly: corpus / premium / priced log-loss | 0.3324 / 0.2810 / 0.3667 | `deploy_and_leak.json · variants.deploy.ll_*` |
| the leak put back, worth on confirm | +0.0038 [+0.0031, +0.0044] | `deploy_and_leak.json · variants.leak-back.delta_confirm` |
| the leak put back, worth on select | +0.0029 [+0.0023, +0.0036] | `deploy_and_leak.json · variants.leak-back.delta_select` |
| the leak put back, worth on prem | +0.0013 [+0.0003, +0.0024] | `deploy_and_leak.json · variants.leak-back.delta_prem` |
| the leak put back, worth on quoted | −0.0007 [−0.0030, +0.0015] | `deploy_and_leak.json · variants.leak-back.delta_quoted` |
| blend weight λ, all priced bouts | 0.15 | `lambda_by_slice.json · global_lambda` |
| λ, distance ≤6 (fit 556 / test 882) | 0.00 | `lambda_by_slice.json · slices[slice=distance ≤6]` |
| λ, distance 8 (fit 444 / test 582) | 0.32 | `lambda_by_slice.json · slices[slice=distance 8]` |
| λ, distance 10 (fit 945 / test 1195) | 0.20 | `lambda_by_slice.json · slices[slice=distance 10]` |
| λ, distance 12 (fit 526 / test 629) | 0.19 | `lambda_by_slice.json · slices[slice=distance 12]` |
| λ, no belt (fit 1467 / test 1946) | 0.13 | `lambda_by_slice.json · slices[slice=no belt]` |
| λ, regional/national belt (fit 311 / test 378) | 0.39 | `lambda_by_slice.json · slices[slice=regional/national belt]` |
| λ, continental/world belt (fit 694 / test 964) | 0.09 | `lambda_by_slice.json · slices[slice=continental/world belt]` |
| λ, thinner man <3 bouts (fit 174 / test 209) | 0.00 | `lambda_by_slice.json · slices[slice=thinner man <3 bouts]` |
| λ, 3–8 (fit 444 / test 657) | 0.06 | `lambda_by_slice.json · slices[slice=3–8]` |
| λ, 8–15 (fit 837 / test 1244) | 0.21 | `lambda_by_slice.json · slices[slice=8–15]` |
| λ, 15+ (fit 1017 / test 1178) | 0.25 | `lambda_by_slice.json · slices[slice=15+]` |
| λ, same country (fit 841 / test 1164) | 0.01 | `lambda_by_slice.json · slices[slice=same country]` |
| λ, different countries (fit 1629 / test 2124) | 0.24 | `lambda_by_slice.json · slices[slice=different countries]` |
| search: candidates screened | 200 | `search.json · n_screened` |
| search: final configuration under five seeds, window A premium | 0.3174–0.3181 | `search.json · seed_band_A_prem` |
| search: median candidate, window A premium | 0.3201 | `search_screen.jsonl · ll_A_prem` |
| search: candidates better than the best seed | 6 | `search_screen.jsonl · ll_A_prem` |
| search: c0063, lead on A / on unseen premium [99%] / survives | +0.0017 / −0.0004 [−0.0012, +0.0004] / no | `search.json · candidates[id=c0063]` |
| search: c0098, lead on A / on unseen premium [99%] / survives | +0.0008 / −0.0014 [−0.0031, +0.0002] / no | `search.json · candidates[id=c0098]` |
| search: c0031, lead on A / on unseen premium [99%] / survives | +0.0008 / −0.0005 [−0.0015, +0.0006] / no | `search.json · candidates[id=c0031]` |
| search: c0034, lead on A / on unseen premium [99%] / survives | +0.0006 / −0.0014 [−0.0026, −0.0001] / no | `search.json · candidates[id=c0034]` |
| search: c0159, lead on A / on unseen premium [99%] / survives | +0.0005 / −0.0038 [−0.0058, −0.0019] / no | `search.json · candidates[id=c0159]` |
| e-value audit P1, fair (3,078 bouts) | 2.5e+08 | `evalue_audit.json · confirmatory.primary.P1.e_fair` |
| e-value audit P2, fair (409 bouts) | 4.4 | `evalue_audit.json · confirmatory.primary.P2.e_fair` |
| e-value audit P3, real prices (3,077 bouts) | 3.9e+08 | `evalue_audit.json · confirmatory.primary.P3.e_real` |
| e-value audit P4, real prices (3,078 bouts) | 1.23e+03 | `evalue_audit.json · confirmatory.primary.P4.e_real` |
| e-value audit C1, the price recalibrated with no model | 0.107 | `evalue_audit.json · confirmatory.control.C1.e_fair` |
| e-value audit: slices at e ≥ 20, of 61 | 30 | `evalue_audit.json · confirmatory.count_e_ge_20` |
| e-value audit B01: the closing favourite is underpriced | 0.0673 | `evalue_audit.json · confirmatory.family.B01.e_fair` |
| e-value audit B03: a fighter on a 10+ win streak, against one who is not, is overpriced | 0.285 | `evalue_audit.json · confirmatory.family.B03.e_fair` |
| e-value audit B05: the home fighter is mispriced (either way) | 2.03 | `evalue_audit.json · confirmatory.family.B05.e_fair` |
| e-value audit: largest of the 16 price biases | 13 | `evalue_audit.json · confirmatory.family.B*` |
| e-value audit null: slices at e ≥ 20, mean / 95th pct | 0.23 / 1 | `evalue_null.json · count_e_ge_20` |
| money at Bet365's true close (kickoff), real e (1,539 bouts) | 1.15 | `evalue_followup.json · close_by_snapshot.kickoff.e_real` |
| money at Bet365's open, model without fight-week data, real e | 1.2e+08 | `evalue_followup.json · open.open_safe_model.e_real` |
| money at the open, price-only controls | 1.23 / 0.246 | `evalue_followup.json · open.controls_no_model` |
| every favourite at Bet365's open, flat | −0.5% [−2.1%, +1.0%] | `evalue_followup.json · open.every_favourite_flat` |
| flat stakes at Bet365's open, λ 0.25 (1,322 bets) | +10.9% [+6.2%, +15.9%] | `evalue_followup.json · open.flat_by_lambda.0.25` |
| money at the open, by year (2023 / 2024 / 2025) | 2.78e+03 / 92.6 / 171 | `evalue_followup.json · open.by_year` |
| money at the open under a simulated null, mean e | 0.20 | `evalue_followup.json · open.null_real.mean_e` |
| line moved towards the model's bets | 69% | `evalue_followup.json · open.clv_of_lambda_0.25_bets` |
| days from Bet365's open to the bell, median | 3 | `evalue_followup.json · open.days_open_to_bout` |
| money at ProBoxingOdds' open, 2023–2025, real e / flat λ 0.25 | 7.0e+06 / +12.2% [+6.8%, +18.2%] | `evalue_followup.json · proboxingodds_open` |
| log-loss 2023–2025, model | 0.3590 | `evalue_followup.json · log_loss.model` |
| log-loss 2023–2025, open | 0.3506 | `evalue_followup.json · log_loss.open` |
| log-loss 2023–2025, open plus model 0.15 | 0.3439 | `evalue_followup.json · log_loss.open_plus_model_0.15` |
| log-loss 2023–2025, open plus model 0.25 | 0.3411 | `evalue_followup.json · log_loss.open_plus_model_0.25` |
| log-loss 2023–2025, open plus model 0.35 | 0.3396 | `evalue_followup.json · log_loss.open_plus_model_0.35` |
| log-loss 2023–2025, close | 0.3379 | `evalue_followup.json · log_loss.close` |
| log-loss 2023–2025, close plus model 0.15 | 0.3336 | `evalue_followup.json · log_loss.close_plus_model_0.15` |
| prereg 2 window A (primary): bouts / real e / fair e | 2,016 / 7.0e+06 / 1.7e+16 | `window_a.json · n_primary, primary` |
| prereg 2 window A (primary): flat ROI range over λ | +18% to +24% | `window_a.json · flat_by_lambda` |
| prereg 2 window A (primary): by year | 2016 1.4 · 2017 39.4 · 2018 8.45 · 2019 1.53e+03 · 2020 1.6 | `window_a.json · by_year` |
| prereg 2 window A (primary) null: mean e / share ≥ 20 | 0.26 / 0.0% | `window_a_null.json · null` |
| prereg 2 window B (secondary): bouts / real e / fair e | 1,545 / 12.5 / 604 | `window_b.json · n_primary, primary` |
| prereg 2 window B (secondary): flat ROI range over λ | −1% to +19% | `window_b.json · flat_by_lambda` |
| prereg 2 window B (secondary): by year | 2021 1.25 · 2022 11.6 · 2023 0.814 | `window_b.json · by_year` |
| prereg 2 window B (secondary) null: mean e / share ≥ 20 | 0.31 / 0.0% | `window_b_null.json · null` |
| calibration slope, corpus | 1.004 | `calibration.json · corpus.calibration_slope` |
| calibration slope, premium | 1.009 | `calibration.json · premium.calibration_slope` |
| calibration slope, quoted | 0.924 | `calibration.json · quoted.calibration_slope` |
| oracle logit factor on priced bouts buys | +0.0010 | `calibration.json · quoted.oracle_gain` |
