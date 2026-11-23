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
