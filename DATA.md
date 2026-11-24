# The shared match table

Our unit of analysis is **one eligible ATP men's main-draw singles match**. Our primary target is **which player wins, conditional on the match being played to completion**. This population leaves out some losses by injured or ill players. For the separate question of who wins a match that starts and ends in retirement, see the [outcome audit](OUTCOMES.md).

## Collect and reproduce

From the repository root, run:

```bash
python3 -m scripts.prepare_data
```

The command downloads or verifies 11 annual CSVs in `data/raw/` and writes `data/processed/model_input.csv` locally. The [manifest](data-manifest.json) pins the source URLs and SHA-256 hashes. The data provider compiles records using ATP updates and other sources. Its terms permit educational research and restrict redistribution of its raw files. Each collaborator regenerates the match-level data locally. [Source, field descriptions and usage terms](https://stats.tennismylife.org/tennis-match-database)

| Population | Matches | Scope |
| --- | ---: | --- |
| Original 2015–2025 ATP annual files | 30,716 | Includes events outside the selected levels and non-completions. |
| Selected event levels, surfaces, IDs and valid rankings | 27,343 | Includes completed matches and reported non-completions. |
| Primary population across all 11 seasons | 26,372 | Completed matches, 768 distinct players and 672 tournament editions. |
| Modeling table generated at this stage | 18,776 | 16,188 training matches from 2016–2022 and 2,588 validation matches from 2023. |

The pinned input files and current exporter produce SHA-256 `39d76da0f5fac7dbe443774ae58e384fc55dc5dc923219f29b5118125aa9722e` for `model_input.csv`. Matching row counts and this digest confirm that two teammates have the same table.

The retained event levels are 250, 500, Masters and Grand Slam, on hard, clay or grass. The 2015 season supplies 2,508 warm-up matches for player history. The 5,088 eligible 2024–2025 matches are reserved for final evaluation. The complete population contains 15,127 hard-court, 8,174 clay-court and 3,071 grass-court matches. Double counting players across seasons or tournaments would inflate a coverage figure, so player and tournament counts above use distinct IDs.

The [loader](scripts/baseline.py) assigns each exclusion to its first failed check. Its counts in order are 3,303 outside event scope, zero additional surface or ID exclusions, 70 invalid or missing rankings, and 971 score exclusions. These counts describe this pinned snapshot. The [outcome audit](OUTCOMES.md) also shows what the ranking requirement removes from each result category. Scope is limited to the included tour levels. Earlier challenger, qualifying or lower-tour wins are outside the players' recorded histories here.

## Coverage of recent draws

The preferred collection window is **2011–2025**, with **2016–2025** as the ten-season comparison. The saved starter notebook uses 2015–2025. For either new window, generate the four-tier main-draw modeling table with the chosen first season, for example `python3 -m scripts.prepare_data --first-season 2011`. The file is `data/processed/model_input_2011.csv`.

| Source series | 2011–2025 raw | 2011–2025 eligible | 2016–2025 raw | 2016–2025 eligible |
| --- | ---: | ---: | ---: | ---: |
| Tour annual singles files | 42,585 | 36,370 | 27,771 | 23,864 |
| Challenger singles main draws | 69,036 | 64,969 | 47,278 | 44,766 |
| Tour-level singles qualifying | 21,636 | 19,907 | 12,411 | 11,728 |

The Tour source contains additional competitions. Its eligible column keeps ATP 250, 500, Masters and Grand Slam main draws. Out of 42,585 records, this removes 4,719 other-level matches (including Davis Cup, Olympics and tour finals), 89 for rankings and 1,407 for incomplete or exceptional endings. A qualifier who advances into the main draw counts normally. The other two series live in separate files. Challenger eligibility uses level C, and qualifying eligibility uses the four selected Tour levels. All three apply the same hard/clay/grass, distinct ID, positive ranking and completion filters. The table counts **available records**, and only Tour main-draw rows are modeling targets. Challenger qualifying, ITF, doubles and women's matches are outside these counts.

**What was cross-checked.** The 2019 Challenger file contains 69 editions and its latest recorded tournament date is June 16. ATP reports 147 events in 2021, eleven fewer than 2019, implying 158 in 2019. A pinned independent results archive contains 3,981 additional Challenger main-draw rows across 85 editions starting after June 16, including the November Ortisei final won by Jannik Sinner. These rows serve as a cross-check and are outside the table above. In 2024 the source contains 192 Challenger editions, while ATP reports 207 tournaments. For a confirmed 2024 gap, the Knoxville draw records Christopher Eubanks defeating Learner Tien in the final. The source has no Knoxville edition (ID 2024-363) or match between these players that season. [Official Knoxville draw](https://www.protennislive.com/posting/2024/363/mds.pdf) [ATP season totals for 2019 and 2021](https://www.atptour.com/en/news/atp-chairman-update-january-2022) · [2024 ATP count](https://www.atptour.com/en/news/challenger-2024-tournament-of-the-year) · [Ortisei final](https://www.atptour.com/en/news/sinner-2019-ortisei-challenger-title) · [Independent 2019 snapshot](https://github.com/Aneeshers/tennis-sackmann-archive/blob/83733587353df8a41f2fd4f516147d5aa83f5a8d/atp/atp_matches_qual_chall_2019.csv)

Official draws provide smaller, match-level checks. All 64 first-round pairings at Wimbledon 2012, all 24 played first-round pairings at Cincinnati 2023 (eight byes), and all 16 at the 2024 Mexico City Challenger match the source after accounting for name variants. Their raw draw record counts were 127, 55 and 31. This spot check establishes coverage for those comparisons only. [Wimbledon draw](https://assets.wimbledon.com/archive/draws/pdfs/draws/2012_MS_A4.pdf) · [Cincinnati draw](https://www.protennislive.com/posting/2023/422/mds.pdf) · [Mexico City draw](https://www.protennislive.com/posting/2024/8375/mds.pdf)

**Why keep draw types separate.** The Challenger file records the 2024 Nottingham grass event (ID 2024-7740) on February 11, the date it also gives a separate hard-court event. The official grass draw ran June 10–16. An uncorrected date could place June results into a player's spring history. For a future history-only feature, first reconcile missing event editions, tournament dates and player IDs, and carry the draw type in the match key because main and qualifying draws reuse match numbers. Until then describe the current feature as **prior covered Tour main-draw results on this surface**. [LTA Nottingham draw](https://www.lta.org.uk/49ca7a/siteassets/events/nottingham/media/atp-challenger-draw---june-11.pdf)

Run `python3 -m scripts.audit_draw_coverage --annual` to reproduce source totals from all 45 season files with local checksum verification. [ATP and Challenger file catalog](https://stats.tennismylife.org/tennis-match-database) · [Supplemental file URLs and checksums](draw-coverage-manifest.json)

## Expand the training history

The historical audit covers 36 annual files from 1990 through 2025. It applies the same event levels, surfaces, player ID checks, rankings and completion rule as the starter analysis. Each option uses its first season for player history, the following seasons through 2022 for training, 2023 for validation and 2024–2025 for final evaluation. The local modeling CSV includes training and validation only. All windows retain the same 2,588 validation matches.

| First season | Raw records through 2025 | Eligible matches, all splits | Training matches | Model CSV rows incl. 2023 |
| --- | ---: | ---: | ---: | ---: |
| 2015 (starter) | 30,716 | 26,372 | 16,188 | 18,776 |
| 2010 | 45,615 | 38,919 | 28,694 | 31,282 |
| 2009 | 48,700 | 41,512 | 31,243 | 33,831 |
| 2000 | 77,953 | 64,896 | 54,509 | 57,097 |
| 1990 | 115,105 | 92,337 | 82,050 | 84,638 |

To build the largest modeling table, run:

```bash
python3 -m scripts.prepare_data --first-season 1990
```

The file is `data/processed/model_input_1990.csv`. Substitute `2009` to build the modern-era comparison. All exports use the existing feature definitions and pinned source snapshots. [Historical file checksums](history-manifest.json) · [1990 source CSV](https://stats.tennismylife.org/data/1990.csv)

**Coverage and comparability.** The full eligible pool covers 2,261 players across 2,262 tournament editions, with 49,752 hard-court, 32,245 clay-court and 10,340 grass-court matches. All 36 headers match and the eligible match keys are unique. Among the selected tiers, the surface filter drops 6,428 historical carpet-court rows. The provider assigns the codes `250`, `500`, `M` and `G` to older editions retrospectively. ATP formally introduced the current tier names in 2009. For example, the source codes Hamburg as `M` in 2008 and `500` in 2009. Use the edition's recorded level. [ATP tour timeline](https://www.atptour.com/-/media/files/media-guide/2026/2026-atp-media-guide-full.pdf) · [Hamburg tournament history](https://hamburgopenatp500.com/en/history)

The longer historical exports are archived alternatives. The active collection work focuses on draw coverage within the recent 10–15 seasons.

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

| Method | Inputs | Question it answers |
| --- | --- | --- |
| Higher-ranked-player rule | `rank_a`, `rank_b` | How often does the favorite win? An equal ranking gives no pick. All 2,588 validation matches have distinct ranks. |
| Rank-only logistic regression | `log_rank_diff` | How much predictive information does ranking supply on its own? |
| Rank and surface logistic regression | `log_rank_diff`, `surface_rate_diff` | How much predictive information does the surface record add to the same model? |

IDs, names and the outcome are for tracing rows or scoring predictions. Use the input columns listed above to fit the models. The two regressions are planned experiments. Both train on 2016–2022 and use the same 2023 validation matches.

## How to read the metrics

- **Accuracy** is correct winner picks divided by scored matches. Higher is better. For probability models, pick A when its estimated probability is at least 0.5. Report the numerator, denominator and percentage. Express a change from 63% to 64% as **+1 percentage point**.
- **Mean log loss** is the average of `-ln(probability assigned to the actual winner)`. Lower is better. If two models give A a 51% and a 90% chance and B wins, both get the winner wrong. Their losses are about 0.713 and 2.303 respectively. This distinguishes a near toss-up from a confident mistake. Report this for the two probability models. The rank rule supplies a pick only. [Metric definitions](https://scikit-learn.org/stable/modules/model_evaluation.html)
- **Compare like with like.** Match predictions by unique ID and use identical eligible rows. Report the change after adding the surface feature in percentage points of accuracy and in log-loss units. A positive accuracy change and a negative log-loss change denote improvement. Show each test season and their pooled result after settings are frozen. Pool per-match values so each match has equal weight. If making a claim of statistical significance, assess paired differences by resampling tournament editions.
- **Keep the target visible.** The result-status audit compares the ranking rule under two definitions of a loss on the 2023 season. That comparison changes the evaluation population. It measures sensitivity to the definition and cannot establish the predictive benefit of a feature or the effect of injuries on a particular player.
