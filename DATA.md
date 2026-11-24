# Men's singles data and evaluation

The [four-feature modeling table](data/processed/model_features.csv) includes ranking, recent overall form, same-surface form and prior head-to-head for 18,774 training and validation matches. Run `python3 -m scripts.prepare_features` to rebuild it offline from the published inventory and starter table. [Starter table](data/processed/model_input.csv) · [Data processing guide (中文)](docs/data-processing.md)

## Combined men's singles inventory

The [processed match CSV](data/processed/mens_singles_2006_2025.csv) is included in the repository. To reproduce it, run `python3 -m scripts.audit_draw_coverage --first-season 2006 --export-inventory`. Each row represents one retained men's singles match. The inventory includes main draws at the four majors, the selected ATP Tour tiers and all recorded Challenger tiers, plus available Tour and major qualifying. Each source uses the same court, positive-ranking, player-ID and completed-score filters. The starter notebook's 63.37% validation benchmark applies to its Tour main-draw population.

**Source credits.** This educational table combines TennisMyLife records with a 2006 supplement from Jeff Sackmann’s ATP data, archived by Aneeshers. The project filters and reconciles matches, aligns player IDs, assigns A/B positions and masks held-out outcomes. TennisMyLife limits use to personal, educational and noncommercial research and restricts redistribution of its raw CSVs. The adapted Sackmann supplement and weekly rank facts used for the four corrected Tour editions carry CC BY-NC-SA 4.0. [TennisMyLife terms](https://stats.tennismylife.org/tennis-match-database) · [Sackmann archive](https://github.com/Aneeshers/tennis-sackmann-archive) · [Supplement license](https://creativecommons.org/licenses/by-nc-sa/4.0/)

| Men's singles category | 2006–2010 added | 2011–2025 | 2006–2025 |
| --- | ---: | ---: | ---: |
| Australian Open, Roland Garros, Wimbledon and US Open main draws | 2,427 | 7,156 | 9,583 |
| Masters 1000, ATP 500 and ATP 250 main draws and their historical equivalents | 10,458 | 29,211 | 39,669 |
| Challenger main draws (all recorded Challenger levels) | 21,301 | 64,969 | 86,270 |
| Tour and major qualifying | 6,561 | 19,906 | 26,467 |
| **All included men's singles** | **40,747** | **121,242** | **161,989** |

Across 20 seasons the inventory contains **135,522 main-draw matches**. The qualifying row includes 8,517 major qualifying matches. We find records for 79 major main draws and 78 major qualifying draws. In 2020 Wimbledon was cancelled and the US Open held its main draw without qualifying. Challenger qualifying is outside this collection. [Wimbledon cancellation](https://www.wimbledon.com/pdf/2020_Compendium.pdf) · [USTA's 2020 format](https://www.usopen.org/en_US/news/articles/2020-06-17/2020_us_open_western_southern_open_to_be_held_in_new_york.html) · [Other known coverage gaps](#coverage-of-recent-draws)

To reproduce the original 15-season export, run `python3 -m scripts.audit_draw_coverage --first-season 2011 --export-inventory`. Those seasons contain 101,336 main-draw matches and 19,906 qualifying matches, including 6,337 in the majors. The ten-season subset from 2016 through 2025 contains 80,356 distinct matches.

**Annual totals after reconciliation.** 2006 supplies the initial history, 2007–2022 contain 125,924 training-period matches, 2023 has 9,288 validation-period matches and 2024–2025 have 19,280 held-out matches. These counts describe collection, before fitting a model. The published 20-season CSV has SHA-256 `6bbd220cdc1a7bc90427f31fb897fd1a4314c2aa4b352f80385b91401fb20613`.

| Season | Matches | Season | Matches |
| --- | ---: | --- | ---: |
| 2006 | 7,497 | 2016 | 8,388 |
| 2007 | 8,278 | 2017 | 8,040 |
| 2008 | 8,552 | 2018 | 8,215 |
| 2009 | 8,365 | 2019 | 6,788 |
| 2010 | 8,055 | 2020 | 3,898 |
| 2011 | 8,008 | 2021 | 7,593 |
| 2012 | 7,987 | 2022 | 8,866 |
| 2013 | 8,169 | 2023 | 9,288 |
| 2014 | 8,163 | 2024 | 9,480 |
| 2015 | 8,559 | 2025 | 9,800 |

**Tournament points and category codes.** Challenger 75 and Challenger 100 both belong to the ATP Challenger Tour. The number indicates the champion's ranking points. The provider uses one level code, `C`, for Challenger editions. As concrete checks, the 2024 Champaign Challenger 75 and Kobe Challenger 100 each contribute 31 eligible main-draw matches in this inventory. Historical Challenger categories changed over time. A count for just the 75 or just the 100 events would require an edition-level crosswalk to official classifications. [ATP points table](https://www.atptour.com/en/news/2024-rankings-points-update) · [Champaign draw](https://www.protennislive.com/posting/2024/637/mds.pdf) · [Kobe draw](https://www.protennislive.com/posting/2024/7397/mds.pdf)

**Reconciliation.** Across all 20 seasons, 162,012 source records pass the filters. Removing 22 duplicates and one spurious first-round entry gives 161,989 matches. In the 2011–2025 subset, 121,246 records pass the filters. Three duplicate pairs have identical source fields except for the match number. They record the same players, score and outcome in the same elimination round. The inventory keeps one copy per pair. The duplicates occur in the 2011 Costa do Sauípe Q1 (one qualifying record), the 2021 Rome R64 and the 2021 Stuttgart R32 (two Tour main-draw records). A fourth entry in the 2015 Båstad file incorrectly gives the Paire–Robredo final a second result in the first round. The official bracket shows Robredo had a bye and Paire faced Markus Eriksson. The inventory keeps the actual final and the actual first-round match. It preserves legitimate meetings across round robins, finals and qualifying stages. [Båstad draw](https://www.protennislive.com/posting/2015/316/mds.pdf)

The unique `match_id` combines the draw source, tournament ID and match number. Player A has the lexicographically smaller ATP ID. A value of 1 in `a_won` means player A won and 0 means player B won. Its tournament date uses the source value with the documented draw-level corrections. Four mixed-date Tour main draws also use a fixed ranking list published before play started. [Tour corrections and ranking archive](scripts/model_event_metadata.py) · [Challenger and qualifying corrections](scripts/inventory_event_metadata.py). Some dates represent the main-draw week even when qualifying was played earlier. Validate suspect dates before calculating chronological features across circuits. The 2024–2025 outcome column is blank for the held-out seasons.

The official Rome and Stuttgart draws confirm the repeated pairings. In Costa do Sauípe the draw also includes Frederico Gil vs. Fabiano de Paula in Q1, missing from the provider file. It is outside the inventory pending an independently verified row with compatible rankings. [Rome draw](https://www.protennislive.com/posting/2021/416/mds.pdf) · [Stuttgart draw](https://www.protennislive.com/posting/2021/321/mds.pdf) · [Costa do Sauípe qualifying draw](https://www.protennislive.com/posting/2011/533/qs.pdf)

### Coverage in the five added seasons

- **2006.** TennisMyLife supplies 144 Challenger editions and no annual Tour qualifying file. Jeff Sackmann's archived results provide the four major qualifying draws and 17 additional Challenger main draws, including Sunrise. The 448 qualifying records yield 437 eligible matches. The 527 Challenger records yield 463. To match players across sources, the supplement checks normalized full names and estimated birth dates against pinned TennisMyLife records, then requires a one-to-one ID mapping. Thirty players have insufficient evidence for a match. Their 39 Challenger records are omitted. The source is credited to Jeff Sackmann under CC BY-NC-SA 4.0. [Supplement code and checksum](scripts/supplemental_draws_2006.py) Other Tour qualifiers took place, including Indian Wells and Rome. Those draws are outside the combined inventory for 2006. [Pinned archive](https://github.com/Aneeshers/tennis-sackmann-archive/blob/83733587353df8a41f2fd4f516147d5aa83f5a8d/atp/atp_matches_qual_chall_2006.csv) · [Sunrise draw](https://www.protennislive.com/posting/2006/2207/mds.pdf) · [Indian Wells qualifying draw](https://www.protennislive.com/posting/2006/404/qs.pdf) · [Rome qualifying draw](https://www.protennislive.com/posting/2006/416/qs.pdf)
- **2007.** Each round of Zagreb qualifying appears twice in the annual file. The 22 pairs agree on all fields except match number. Nineteen pairs pass the eligibility checks. The export keeps one copy of each eligible match, then verifies equality against the pinned rows. [Official Zagreb qualifying draw](https://www.protennislive.com/posting/2007/451/qs.pdf)
- **2010.** The qualifying file omits both tier and surface for 356 records from 11 events. For those editions, the corresponding main draw has one consistent tier and surface and matches the qualifying event ID, name and reported date. Filling the two event fields restores 319 eligible matches, including 110 from Wimbledon. Of the other 37 records, 29 fail the ranking check and eight fail the completion check. [Recovery code](scripts/qualifying_levels.py) · [Wimbledon qualifying draw](https://assets.wimbledon.com/archive/draws/pdfs/draws/2010_QS_A4.pdf) · [Hamburg qualifying draw](https://www.protennislive.com/posting/2010/414/qs.pdf)

Before 2009, the provider labels older Tour tiers retrospectively as 250 or 500. The three-surface restriction also excludes historical indoor carpet matches. In the 2007 Australian Open qualifying, 2008 Las Vegas qualifying, 2007 Belo Horizonte and 2008 Cremona draws, four eligible source score strings are inconsistent with the recorded winner's sets. The draws confirm the recorded players advanced. The inventory retains the winners and supplies no set-score features. [Australian Open qualifying](https://www.protennislive.com/posting/2007/580/qs.pdf) · [Las Vegas qualifying](https://www.protennislive.com/posting/2008/433/qs.pdf) · [Belo Horizonte](https://www.protennislive.com/posting/2007/3633/mds.pdf) · [Cremona](https://www.protennislive.com/posting/2008/3812/mds.pdf)

## Coverage of recent draws

The collection window is **2006–2025**. The following source-level comparisons document the original 15- and ten-season windows. The saved starter notebook uses 2015–2025. Generate a four-tier Tour main-draw modeling table for a chosen first season, for example `python3 -m scripts.prepare_data --first-season 2011`. The file is `data/processed/model_input_2011.csv`.

| Source series | 2011–2025 raw | 2011–2025 eligible | 2016–2025 raw | 2016–2025 eligible |
| --- | ---: | ---: | ---: | ---: |
| Tour annual singles files | 42,585 | 36,370 | 27,771 | 23,864 |
| Challenger singles main draws | 69,036 | 64,969 | 47,278 | 44,766 |
| Tour-level singles qualifying | 21,636 | 19,907 | 12,411 | 11,728 |

The Tour source contains additional competitions. Its eligible column keeps ATP 250, 500, Masters and Grand Slam main draws. Out of 42,585 records, this removes 4,719 other-level matches (including Davis Cup, Olympics and tour finals), 89 for rankings and 1,407 for incomplete or exceptional endings. A qualifier who advances into the main draw counts normally. The other two series live in separate files. Challenger eligibility uses level C, and qualifying eligibility uses the four selected Tour levels. All three apply the same hard/clay/grass, distinct ID, positive ranking and completion filters. The table counts **available records**, and the original modeling notebook evaluates the Tour main-draw subset. Challenger qualifying, ITF, doubles and women's matches are outside these counts.

**What was cross-checked.** The 2019 Challenger file contains 69 editions and its latest recorded tournament date is June 16. ATP reports 147 events in 2021, eleven fewer than 2019, implying 158 in 2019. A pinned independent results archive contains 3,981 additional Challenger main-draw rows across 85 editions starting after June 16, including the November Ortisei final won by Jannik Sinner. These rows serve as a cross-check and are outside the table above. In 2024 the source contains 192 Challenger editions, while ATP reports 207 tournaments. For a confirmed 2024 gap, the Knoxville draw records Christopher Eubanks defeating Learner Tien in the final. The source has no Knoxville edition (ID 2024-363) or match between these players that season. [Official Knoxville draw](https://www.protennislive.com/posting/2024/363/mds.pdf) [ATP season totals for 2019 and 2021](https://www.atptour.com/en/news/atp-chairman-update-january-2022) · [2024 ATP count](https://www.atptour.com/en/news/challenger-2024-tournament-of-the-year) · [Ortisei final](https://www.atptour.com/en/news/sinner-2019-ortisei-challenger-title) · [Independent 2019 snapshot](https://github.com/Aneeshers/tennis-sackmann-archive/blob/83733587353df8a41f2fd4f516147d5aa83f5a8d/atp/atp_matches_qual_chall_2019.csv)

Official draws provide smaller, match-level checks. All 64 first-round pairings at Wimbledon 2012, all 24 played first-round pairings at Cincinnati 2023 (eight byes), and all 16 at the 2024 Mexico City Challenger match the source after accounting for name variants. Their raw draw record counts were 127, 55 and 31. This spot check establishes coverage for those comparisons only. [Wimbledon draw](https://assets.wimbledon.com/archive/draws/pdfs/draws/2012_MS_A4.pdf) · [Cincinnati draw](https://www.protennislive.com/posting/2023/422/mds.pdf) · [Mexico City draw](https://www.protennislive.com/posting/2024/8375/mds.pdf)

**Why keep draw types separate.** The Challenger file records the 2024 Nottingham grass event (ID 2024-7740) on February 11, the date it also gives a separate hard-court event. The published inventory now uses the official June 10 start for the grass draw. The edition's source rankings still differ across rounds. For a future history-only feature, first reconcile missing event editions, tournament dates and player IDs, and carry the draw type in the match key because main and qualifying draws reuse match numbers. Until then describe the current feature as **prior covered Tour main-draw results on this surface**. [LTA Nottingham draw](https://www.lta.org.uk/49ca7a/siteassets/events/nottingham/media/atp-challenger-draw---june-11.pdf)

Run `python3 -m scripts.audit_draw_coverage --first-season 2006 --annual --export-inventory` to reproduce the 20-season source counts and reconciled export. All input files are checked against pinned SHA-256 hashes, including the extra 2005 Tour file used only for identity matching. [ATP and Challenger file catalog](https://stats.tennismylife.org/tennis-match-database) · [Supplemental file URLs and checksums](draw-coverage-manifest.json)

## Starter notebook match table

Our unit of analysis is **one eligible ATP men's main-draw singles match**. Our primary target is **which player wins, conditional on the match being played to completion**. This population leaves out some losses by injured or ill players. For the separate question of who wins a match that starts and ends in retirement, see the [outcome audit](OUTCOMES.md).

### Collect and reproduce

From the repository root, run:

```bash
python3 -m scripts.prepare_data
```

The command downloads or verifies 11 annual CSVs in `data/raw/` and writes `data/processed/model_input.csv` locally. The [manifest](data-manifest.json) pins the source URLs and SHA-256 hashes. The data provider compiles records using ATP updates and other sources. Its terms permit educational research and restrict redistribution of its raw files. The processed model table is available directly in this repository. [Source, field descriptions and usage terms](https://stats.tennismylife.org/tennis-match-database)

| Population | Matches | Scope |
| --- | ---: | --- |
| Original 2015–2025 ATP annual files | 30,716 | Includes events outside the selected levels and non-completions. |
| Selected event levels, surfaces, IDs and valid rankings | 27,343 | Includes completed matches and reported non-completions. |
| Primary population across all 11 seasons | 26,369 | Completed matches, 768 distinct players and 672 tournament editions. |
| Modeling table generated at this stage | 18,774 | 16,186 training matches from 2016–2022 and 2,588 validation matches from 2023. |

The pinned input files and current exporter produce SHA-256 `b61a52744fd337eede9c5cc03203287c1a284df847ace3908cfb41882ea3fadf` for `model_input.csv`. Matching row counts and this digest confirm that two teammates have the same table. The four-feature export, `model_features.csv`, contains the same 18,774 match IDs and has SHA-256 `6291a9b51f5aac5d12cf1611c1d89df34350a4f2111d20997ab007c496207d27`. Its six new count columns give each player’s recent wins and matches, the number of prior meetings, and A’s wins in those meetings.

The retained event levels are 250, 500, Masters and Grand Slam, on hard, clay or grass. The 2015 season supplies 2,507 warm-up matches for player history. The 5,088 eligible 2024–2025 matches are reserved for final evaluation. The complete population contains 15,127 hard-court, 8,172 clay-court and 3,070 grass-court matches. Double counting players across seasons or tournaments would inflate a coverage figure, so player and tournament counts above use distinct IDs.

The [loader](scripts/baseline.py) assigns each exclusion to its first failed check. Its counts in order are 3,303 outside event scope, zero additional surface or ID exclusions, 70 invalid or missing rankings, and 971 score exclusions. A further two duplicated knockout rows and one spurious first-round row are removed after the eligibility filters. The [outcome audit](OUTCOMES.md) also shows what the ranking requirement removes from each result category. Scope is limited to the included tour levels. Earlier challenger, qualifying or lower-tour wins are outside the players' recorded histories here.

## Expand the training history

The historical audit covers 36 annual files from 1990 through 2025. It applies the same event levels, surfaces, player ID checks, rankings and completion rule as the starter analysis. Each option uses its first season for player history, the following seasons through 2022 for training, 2023 for validation and 2024–2025 for final evaluation. The local modeling CSV includes training and validation only. All windows retain the same 2,588 validation matches.

| First season | Raw records through 2025 | Eligible matches, all splits | Training matches | Model CSV rows incl. 2023 |
| --- | ---: | ---: | ---: | ---: |
| 2015 (starter) | 30,716 | 26,369 | 16,186 | 18,774 |
| 2010 | 45,615 | 38,916 | 28,691 | 31,279 |
| 2009 | 48,700 | 41,509 | 31,240 | 33,828 |
| 2000 | 77,953 | 64,893 | 54,506 | 57,094 |
| 1990 | 115,105 | 92,334 | 82,047 | 84,635 |

To build the largest modeling table, run:

```bash
python3 -m scripts.prepare_data --first-season 1990
```

The file is `data/processed/model_input_1990.csv`. Substitute `2009` to build the modern-era comparison. All exports use the existing feature definitions and pinned source snapshots. [Historical file checksums](history-manifest.json) · [1990 source CSV](https://stats.tennismylife.org/data/1990.csv)

**Coverage and comparability.** The full eligible pool covers 2,261 players across 2,262 tournament editions, with 49,752 hard-court, 32,243 clay-court and 10,339 grass-court matches. All 36 headers match and the eligible match keys are unique. Among the selected tiers, the surface filter drops 6,428 historical carpet-court rows. The provider assigns the codes `250`, `500`, `M` and `G` to older editions retrospectively. ATP formally introduced the current tier names in 2009. For example, the source codes Hamburg as `M` in 2008 and `500` in 2009. Use the edition's recorded level. [ATP tour timeline](https://www.atptour.com/-/media/files/media-guide/2026/2026-atp-media-guide-full.pdf) · [Hamburg tournament history](https://hamburgopenatp500.com/en/history)

The longer historical Tour-only exports are archived alternatives. The combined inventory covers the 20 seasons from 2006 through 2025.

## Read one row

| CSV columns | Meaning and use |
| --- | --- |
| `match_id`, `tourney_id`, `match_num` | Unique key is tournament ID and match number joined with a colon. Join model predictions on this key. |
| `season`, `split`, `tourney_date`, `round`, `surface` | Use the annual file's season to reproduce splits. The date is the tournament start week and can fall in late December of the previous year. |
| `player_a_id`, `player_b_id`, `player_a_name`, `player_b_name` | The lexicographically smaller player ID takes position A, regardless of who won. Names are for explaining examples. |
| `rank_a`, `rank_b`, `log_rank_diff` | A smaller rank number is better. The model feature is `log(rank_b) - log(rank_a)`, positive when A is better ranked. |
| `surface_matches_a/b`, `surface_wins_a/b`, `latest_history_date_a/b` | For each player, eligible previous matches and wins on this match's surface, and the latest eligible tournament start date. The CSV has a separate column for each side. |
| `surface_rate_a/b`, `surface_rate_diff` | Each estimate is `(wins + 1) / (matches + 2)`. Zero prior matches gives 0.5 and an empty latest-history date. Subtract B's estimate from A's. |
| `a_won` | Outcome label, 1 if A won and 0 if B won. |

If the current tournament starts at T, previous events qualify from T − 364 days **through** T − 21 days, inclusive. Use only events on the same surface that meet the primary completion rule. This conservative cutoff avoids using outcomes from the same or overlapping tournaments and sacrifices the latest three weeks of form. The [notebook](analysis.ipynb) checks a real match against the underlying records.

## What goes into each comparison

The higher-ranked-player rule gives a pick when ranks differ. All 2,588 validation matches have distinct ranks. Each logistic regression estimates the probability that A wins. The four feature groups each contribute one predictor.

| Group | Column | Definition for a match with tournament start date T |
| --- | --- | --- |
| Ranking | `log_rank_diff` (published) | Log rank of B minus log rank of A. |
| Recent overall form | `overall_rate_diff` (published) | Across all surfaces and opponents, use eligible matches from T−90 through T−21 days inclusive. Compute each player's smoothed win rate and subtract B's from A's. |
| Same-surface form | `surface_rate_diff` (published) | On the current surface, use eligible matches from T−364 through T−21 days inclusive. Subtract B's smoothed win rate from A's. |
| Prior head-to-head | `h2h_rate_diff` (published) | Across all surfaces, count direct meetings between these players from the 2006 season through T−21 days inclusive. Compute twice A's smoothed win rate against B minus one. Zero prior meetings gives zero. |

For each rate use `(wins + 1) / (matches + 2)`. Use only eligible Tour and major main draws (`source == "tour"`) to compute the histories. For each target in the baseline table, look up `"tour:" + match_id` in the inventory and verify both player IDs. Calculate each new value using earlier results available by its cutoff, then left-join the two derived columns back to the target rows. Preserve the A/B orientation and all 18,774 matches, including pairs with no previous meeting. IDs and names identify the participants, and `a_won` supplies the outcome.

Use identical 2016–2022 training rows and 2023 validation rows for all eight prespecified logistic regressions. Each includes ranking. The optional groups are (1) none, (2) overall, (3) surface, (4) head-to-head, (5) overall and surface, (6) overall and head-to-head, (7) surface and head-to-head, and (8) all three. Compare the rank-only regression with the full four-group regression as planned. Report all eight validation results and select the subset with the lowest 2023 mean log loss. Freeze the choices before evaluating 2024–2025. Score a separate overall Elo baseline on the same validation matches using the same historical cutoff. The fits and probability comparisons are pending.

## How to read the metrics

Use the same [probability scorer](scripts/metrics.py) for rank-only logistic regression, each feature combination and Elo. Pass every outcome in the evaluation split and one full-precision probability of an A win per match ID. The scorer checks that both sets of IDs match and rejects duplicates. For example, if `validation` is the modeling DataFrame filtered to 2023 and `forecasts` contains the model's output:

```python
from scripts.metrics import score_probabilities

scores = score_probabilities(
    zip(validation["match_id"], validation["a_won"]),
    zip(forecasts["match_id"], forecasts["p_a"]),
)
```

The result includes `matches`, `correct`, `accuracy` (a fraction) and `mean_log_loss`. The scorer uses the reported probabilities directly. Giving the eventual winner exactly zero probability incurs infinite log loss. For a numerical check, a model assigning 0.5 to each player in every match has mean log loss `ln(2) ≈ 0.6931` for any set of outcomes.

- **Accuracy** is correct winner picks divided by scored matches. Higher is better. For probability models, pick A when its estimated probability is at least 0.5. Report the numerator, denominator and percentage. Express a change from 63% to 64% as **+1 percentage point**.
- **Mean log loss** is the average of `-ln(probability assigned to the actual winner)`. Lower is better. If two models give A a 51% and a 90% chance and B wins, both get the winner wrong. Their losses are about 0.713 and 2.303 respectively. This distinguishes a near toss-up from a confident mistake. Report this for each probability model. The rank rule supplies a pick only. [Metric definitions](https://scikit-learn.org/stable/modules/model_evaluation.html)
- **Compare like with like.** Match predictions by unique ID and use identical eligible rows. Report each subset's difference from rank-only in accuracy percentage points and log-loss units. A positive accuracy change and a negative log-loss change denote improvement. Show each test season and their pooled result after settings are frozen. Pool per-match values so each match has equal weight. If making a claim of statistical significance, assess paired differences by resampling tournament editions.
- **Keep the target visible.** The result-status audit compares the ranking rule under two definitions of a loss on the 2023 season. That comparison changes the evaluation population. It measures sensitivity to the definition and cannot establish the predictive benefit of a feature or the effect of injuries on a particular player.
