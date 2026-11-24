# How retirements affect the ranking benchmark

Adding matches that ended in retirement lowered the higher-ranked player's 2023 validation accuracy by **0.15 percentage points**, from **63.37%** to **63.22%**. The main experiment predicts the winner conditional on both players completing the match. This sensitivity check asks how the ranking rule fares when the recorded winner of a retirement also counts.

## Which match endings occur?

This source audit counts reported records before the later draw reconciliation. Three records in its completed group were removed from the 2015–2025 modeling population. They are outside the 2023 validation season. [Cleaning and final populations](data-processing.md)

The 11 pinned ATP men's season files for 2015–2025 contain 30,716 records. The first row below includes only ATP 250, ATP 500, Masters and Grand Slam matches on hard, clay or grass, with two distinct player IDs. The second row also requires a positive ranking for both players. Season follows the provider's file year. [Data and field definitions](https://stats.tennismylife.org/tennis-match-database) · [Pinned files](../data-manifest.json)

| Sample | All records | Completed | Retirement (RET) | Walkover (W/O) | Default (DEF) | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2015–2025 before ranking filter | 27,413 | 26,440 | 787 | 179 | 7 | 0 |
| 2015–2025 with valid rankings | 27,343 | 26,372 | 785 | 179 | 7 | 0 |
| 2023 validation with valid rankings | 2,676 | 2,588 | 68 | 19 | 1 | 0 |

The ranking filter removes 68 completed matches and two retirements. Every selected W/O appears as a standalone marker. Every selected RET and DEF follows a partial score. All 26,440 unmarked scores parse as sets, end in a set won by the recorded winner and give that player the required number of sets. The script lists unexpected score forms for review.

The ATP glossary defines a retirement as a player stopping after play began because of illness or injury. A walkover occurs before play begins. A default is a separate disciplinary outcome. The CSV records the ending in the score field and provides no medical reason or injury date. A player who finished while impaired would still appear under “Completed.” A medical condition or injury rate cannot be inferred from these counts. [ATP Official Rulebook, Exhibit K, “Draw Results”](https://www.atptour.com/-/media/files/rulebook/2025/2025-rulebook-chapter-10_exhibits_19aug.pdf)

Across the ranked sample, reported retirements account for 785/27,343 records (2.87%). By court surface the shares are 492/15,733 (3.13%) on hard, 215/8,441 (2.55%) on clay and 78/3,169 (2.46%) on grass. Each denominator includes all recorded endings on that surface. These descriptive shares pool seasons, players and events.

## Validation check

| 2023 outcomes scored | Higher-ranked player won | Accuracy |
| --- | ---: | ---: |
| Completed matches, main benchmark | 1,640 / 2,588 | 63.37% |
| Retirement subset | 39 / 68 | 57.35% |
| Completed matches plus retirements | 1,679 / 2,656 | 63.22% |

There were no equal rankings in either scored group. In both versions the 19 walkovers and one default are outside the scoring denominator. This one-season check uses the same pre-match ranking rule throughout. It does not establish how including retirements would affect the two fitted models or whether an injured player would have won a completed match.

## If the research question expands

A player who retires repeatedly accumulates recorded losses in matches that started. Excluding those matches selects for completion, so the primary benchmark describes a narrower population than a forecast of who advances in a draw. For a later study of retirements, a possible pre-match feature is a player's prior RET losses divided by prior starts, accompanied by the exposure count. A simple count gives more opportunities to players who play more often. Compute both values strictly from earlier tournaments and keep W/O separate. A player's current RET flag is learned after the match begins and would reveal information unavailable when predicting. This remains a candidate follow-up.

From the repository root, run the command below. The script verifies each season file against its recorded SHA-256 hash. For an existing cache, add `--cache-dir /path/to/season/files`. [Audit code](../scripts/audit_outcomes.py)

```bash
python3 scripts/audit_outcomes.py
```
