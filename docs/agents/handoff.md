# Team handoff

**Current focus.** Use the published modeling rows and match inventory to compare rankings with three sources of historical information: recent overall form, same-surface results and prior meetings between the opponents. Model targets and player histories use eligible ATP Tour and major main draws. The 161,989-match inventory also covers Challenger main draws and available Tour/major qualifying. [Inventory and coverage](../data.md#combined-mens-singles-inventory) · [Data and metric definitions](../data.md) · [Retirement sensitivity](../outcomes.md)

## What can be reproduced in this version

Start a modeling branch from the latest `main`. Read the [four-feature modeling CSV](../../data/processed/model_features.csv) to fit the planned models. It contains all 18,774 training and validation matches. For the worked Cincinnati final and the offline rebuild, see the [data processing guide](../data-processing.md).

| Evidence | Where to look |
| --- | --- |
| The loader retains 26,369 eligible matches across 11 seasons. The higher-ranked player won **1,640 / 2,588** validation matches in 2023 (**63.37%**). | Run `python3 scripts/baseline.py` from the repo root. [Loader and ranking rule](../../scripts/baseline.py) |
| Training figures show how often the favorite wins as the rank gap grows, and how much prior surface history is available. With the adopted window, 77.5% of grass-court player appearances have fewer than five eligible prior matches, compared with 29.4% on clay and 13.8% on hard courts. | [Executed notebook, section 2](../../analysis.ipynb) |
| A shared feature table contains 18,774 training and validation matches. An independent source-row calculation replicates both players' prior hard-court records for the 2023 Cincinnati final. | Run `python3 -m scripts.prepare_data`. [Schema and population](../data.md) · [Worked match](../../analysis.ipynb) |
| The 20-season inventory contains 49,252 Tour and major main-draw matches, 86,270 Challenger main-draw matches and 26,467 qualifying matches. The five added seasons contribute 40,747 distinct matches. | Run `python3 -m scripts.audit_draw_coverage --first-season 2006 --export-inventory`. [Annual counts and coverage gaps](../data.md#combined-mens-singles-inventory) |
| The archived optional historical pool covers 92,334 eligible matches in 1990–2025. Of these, 82,047 are training matches and 2,588 validation matches. | Run `python3 -m scripts.prepare_data --first-season 1990`. [Historical evidence and comparison design](../data.md#expand-the-training-history) |

The logistic models and overall Elo benchmark still need to be run. Once their settings are frozen, the evaluation teammate can score the 5,088 held-out matches across 2024–2025. The result-based literature discussion, submission HTML and video follow.

## What the next agent should deliver

Use the shared scorer for every probability model. Supply probabilities for player A along with their match IDs and compare them with the full split. [Example call and metric definitions](../data.md#how-to-read-the-metrics)

1. Load the four predictors and the separate A-win label. Check the Cincinnati example in the data guide, including player orientation and the T−21 cutoff. The published table includes counts for interpretation. [Feature definitions](../data.md#what-goes-into-each-comparison)
2. Fit logistic regression on the same 16,186 training matches for each of the eight prespecified subsets. Each includes rankings. Compare rank-only with the full four-group model, and report accuracy and mean log loss for every subset on the same 2,588 validation matches. Include the ranking rule's accuracy as a benchmark. Score an overall Elo probability baseline on the same matches and historical cutoff. Record the initialization, update settings and order for matches sharing a date.
3. Select the subset with the lowest 2023 mean log loss. Share the notebook, rerun command and frozen feature and model settings for the teammate evaluating 2024–2025. [Modeling issue](https://github.com/Jerry-111/atp-match-prediction/issues/1) · [Current research design](../project-plan.zh.md)

## Research ideas and decisions

**Adopted question.** Which information improves match-win probabilities beyond ATP rankings? Compare recent overall form, same-surface form and the opponents' prior head-to-head results under the same fitting and evaluation rules. In 16,186 training matches, 3,570 have a recorded prior meeting in the 52-week window. Extending the lookback to 2006 with the same 21-day lag raises coverage to 7,597. These are coverage counts. The 2023 log-loss comparison will measure predictive value. [Research plan](../project-plan.zh.md)

**Ideas to revisit.** These are candidates for a later decision. Each has a concrete first check.

| Idea | Why it might matter | Smallest useful check |
| --- | --- | --- |
| Look at close rankings | Rank alone gives less separation for players who are close together. | Compare rank-only and all four groups on validation matches with a rank gap of at most 10. Include the match count. |
| Try a longer history on grass | The training figure shows many players have little grass-court history. | Compare coverage under 52 and 104 weeks on seasons with adequate warm-up. Check the change on validation before widening the main experiment. |
| Historical serving and returning | Point totals distinguish how players win, even when their win rates are similar. | Once the main comparisons are complete, compare all four features against those four plus past serve- and return-points-won rates. Use T−364 through T−21 across surfaces, aggregate points by opportunities, and keep low-history players. |

**Decision history**

| Date | Change and reason | Consequence |
| --- | --- | --- |
| 2026-11-23 | Adopted overall Elo as an independent benchmark to account for opponent strength. Historical serve and return rates are an optional extension. | The modeling teammate can start from the published four-feature table, fit eight subsets and compare Elo on the same rows. |
| 2026-11-23 | Expanded the comparison to rankings, recent overall form, same-surface form and prior head-to-head meetings. | The existing two-feature table supplies the baseline. Derive two more features and compare all eight rank-containing subsets on the existing target matches. |
| 2026-11-23 | Extended the combined collection from 15 to 20 seasons. The 2006 backfill uses four major qualifying draws and 17 missing Challenger editions with players matched to ATP IDs. | The 20-season inventory is ready for data work. The starter notebook still reports its original Tour-only experiment. |
| 2026-11-23 | The first modeling dataset focused on rankings and same-surface history. | The initial scope and ranking benchmark landed in [11d91ab](https://github.com/Jerry-111/atp-match-prediction/commit/11d91abad3d155f10cad6b7cadb14a65d5b5d2cf). |
