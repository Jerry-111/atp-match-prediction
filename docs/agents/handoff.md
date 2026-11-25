# Team handoff

**Current focus.** Independently reproduce the completed modeling/validation experiment, then evaluate frozen models on the still-unopened 2024–2025 test seasons. The user authorized a broader strong-model comparison on 2026-11-24. [Modeling protocol and commands](../modeling.md) · [Validation table](../../results/modeling/validation_comparison.csv) · [Frozen model choices](../../results/modeling/frozen_selection.json) · [Research evidence](../research/sota_evidence.md)

**Branch:** `codex/strong-model-validation`. This extends the original eight logistic subsets with historical strength, multiscale form, technical statistics and boosting. It does not establish a universal SOTA result.

## What can be reproduced in this version

Use the modeling branch above until this work merges into `main`. The [four-feature modeling CSV](../../data/processed/model_features.csv) contains all 18,774 training and validation matches. For the worked Cincinnati final and the offline rebuild, see the [data processing guide](../data-processing.md). The additional model inputs are rebuilt by the modeling command.

| Evidence | Where to look |
| --- | --- |
| The loader retains 26,369 eligible matches across 11 seasons. The higher-ranked player won **1,640 / 2,588** validation matches in 2023 (**63.37%**). | Run `python3 scripts/baseline.py` from the repo root. [Loader and ranking rule](../../scripts/baseline.py) |
| Training figures show how often the favorite wins as the rank gap grows, and how much prior surface history is available. With the adopted window, 77.5% of grass-court player appearances have fewer than five eligible prior matches, compared with 29.4% on clay and 13.8% on hard courts. | [Executed notebook, section 2](../../analysis.ipynb) |
| A shared feature table contains 18,774 training and validation matches. An independent source-row calculation replicates both players' prior hard-court records for the 2023 Cincinnati final. | Run `python3 -m scripts.prepare_data`. [Schema and population](../data.md) · [Worked match](../../analysis.ipynb) |
| The 20-season inventory contains 49,252 Tour and major main-draw matches, 86,270 Challenger main-draw matches and 26,467 qualifying matches. The five added seasons contribute 40,747 distinct matches. | Run `python3 -m scripts.audit_draw_coverage --first-season 2006 --export-inventory`. [Annual counts and coverage gaps](../data.md#combined-mens-singles-inventory) |
| The archived optional historical pool covers 92,334 eligible matches in 1990–2025. Of these, 82,047 are training matches and 2,588 validation matches. | Run `python3 -m scripts.prepare_data --first-season 1990`. [Historical evidence and comparison design](../data.md#expand-the-training-history) |

The modeling run is complete: 72 learned configurations plus three raw Elo settings were compared in forward training-period folds. Every reported model uses the same 2,588 validation matches. Ranking logistic log loss is **0.62980** (63.37% accuracy); the advanced recommendation frozen before 2023 is the convex ensemble at **0.61153** (65.07%), a **2.90%** relative loss reduction. Its paired date-block exploratory interval for absolute loss reduction is **[0.01233, 0.02458]**. This interval does not account for the full model search or cross-date player dependence.

The lowest 2023 loss is the richer regularized linear model at **0.61056** (65.26%). Preserve the distinction between this observed validation ordering and the pre-2023 recommendation. The prescribed eight-subset winner is ranking + recent overall form + H2H, **0.62891**; all four original groups score **0.63021**. Original feature additions alone do not show a clear improvement over rankings. Matched advanced linear ablations increase loss by **0.00860** without explicit Elo features, **0.00449** without point-statistic features and **0.00286** without rank features. The Elo ablation retains opponent-adjusted residual form, so it is not removal of every Elo-derived quantity.

The evaluation teammate can now score the 5,088 held-out matches across 2024–2025 using the frozen protocol. No test-season outcome has been evaluated. Video production remains a later team task.

## What the next agent should deliver

Use the shared scorer for every probability model. Supply probabilities for player A along with their match IDs and compare them with the full split. [Example call and metric definitions](../data.md#how-to-read-the-metrics)

1. Install `requirements-modeling.txt`; run `python -m scripts.train_models --with-serve` and `python -m scripts.model_report`. Confirm source hashes, selected settings and saved validation probabilities. Read the executed modeling section in `analysis.ipynb` and the dedicated `modeling.ipynb`.
2. Verify the Cincinnati worked match and the T−21 cutoff. Extend the explicit development-only feature gates for final evaluation without changing feature definitions or model parameters. Historical updates may incorporate earlier test results only when available under the same cutoff; do not refit coefficients on 2023 or test outcomes.
3. Evaluate ranking, original full/subset, raw Elo, the frozen advanced ensemble and its best-single comparator on identical 2024–2025 rows. Report per-year and combined metrics, calibration and paired uncertainty. Label every test result explicitly and record any protocol deviation before scoring. [Modeling issue](https://github.com/Jerry-111/atp-match-prediction/issues/1) · [Current research design](../project-plan.zh.md)

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
| 2026-11-24 | User authorized broad optimization toward stronger predictions. Preserved the original eight subsets and unopened test; added temporal tuning, richer history/point features, symmetric boosting and an OOF ensemble. | Validation experiment complete. Freeze artifacts and final-test handoff replace the previous pending-modeling task; see the modeling protocol and full results. |
| 2026-11-23 | Adopted overall Elo as an independent benchmark to account for opponent strength. Historical serve and return rates are an optional extension. | The modeling teammate can start from the published four-feature table, fit eight subsets and compare Elo on the same rows. |
| 2026-11-23 | Expanded the comparison to rankings, recent overall form, same-surface form and prior head-to-head meetings. | The existing two-feature table supplies the baseline. Derive two more features and compare all eight rank-containing subsets on the existing target matches. |
| 2026-11-23 | Extended the combined collection from 15 to 20 seasons. The 2006 backfill uses four major qualifying draws and 17 missing Challenger editions with players matched to ATP IDs. | The 20-season inventory is ready for data work. The starter notebook still reports its original Tour-only experiment. |
| 2026-11-23 | The first modeling dataset focused on rankings and same-surface history. | The initial scope and ranking benchmark landed in [11d91ab](https://github.com/Jerry-111/atp-match-prediction/commit/11d91abad3d155f10cad6b7cadb14a65d5b5d2cf). |
