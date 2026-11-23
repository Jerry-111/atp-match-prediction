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

## Expand the training history

The historical audit covers 36 annual files from 1990 through 2025. It applies the same event levels, surfaces, player ID checks, rankings and completion rule as the starter analysis. Each option uses its first season for player history, the following seasons through 2022 for training, 2023 for validation and 2024–2025 for final evaluation. The local modeling CSV includes training and validation only. All 2,588 validation rows were checked field by field across the five exports and match exactly.

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

**How to use the extra data.** A sensible first comparison uses 2009 for warm-up and 2010–2022 for training. Repeat with 1990 as warm-up and 1991–2022 for training, against the same 2,588 validation matches from 2023. Keep feature windows, models, settings and evaluation rows comparable. Report accuracy and log loss for each model and training span. This tests whether older matches help the model predict recent tennis. The longer history adds training examples. Each player's form at prediction time still draws on the prior 52 weeks. The existing 2024–2025 holdout remains reserved for final evaluation after choosing the training span.

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
