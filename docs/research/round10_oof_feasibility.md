# Round 10: primitive-expert OOF feasibility and temporal audit

Read-only audit, 2026-11-28. This reviews the conditional-stacking proposal in
[the learning ideas](round10_learning_ideas.md); it is not a fitted experiment
or an approved final menu. No model was fitted, no new 2023 model ranking or
error-based expert selection was performed, and 2024–2025 were not evaluated.
The already completed Round9 neural audit is a separate task.

**Conclusion:** the existing caches support a small forward meta experiment
without generating older neural forecasts. The first honest meta test is
**2021**, not 2020. Four named primitives are available. After this availability
review, the coordinator fixed those four for protocol preparation and synthetic
tests; the optional point/tree alternatives listed below are **deferred**, not
additional search entries. Empirical fitting still requires the separate exact
protocol review.

## Availability and direct label dependencies

All five source OOF files below contain the same **6,130 unique match IDs**, in
the exact target order, with identical `season` and `a_won`: 1,214 in 2020,
2,388 in 2021, and 2,528 in 2022. All named probabilities are finite and in
[0,1]. Their file hashes match the respective independent artifact audits.
This check did not read the branches' 2023 prediction tables.

| Proposed role | Exact primitive column and source | Dependence of predictions for year Y | Saved models / evidence |
| --- | --- | --- | --- |
| Core additive expert | R4 `add_age_spline_C0.01` | Coefficients, spline scales and context transforms fit eligible years before Y. No probability calibration. | Final model saved. Historical fold estimators were not saved; chronology is supported by the runner, deterministic masks, cache binding and earlier audit, not direct fold-weight reload. |
| Core linear expert | R5 `r5_linear_logistic_C0.003` | Native logistic probability; feature map and coefficients fit eligible years before Y. The class sets calibration slope to 1 for this logistic recipe. | Final model saved. Historical fold estimators were not saved; same evidence limit as R4. |
| Core tabular neural expert | R9 neural `realmlp_td_s` | Three fixed seeds averaged equally. For each outer year, epochs are selected on a purged last-season tail inside its training data, then weights and preprocessing refit on the full earlier-year training population. No outer-year label chooses epochs. | All fold/final seed weights and preprocessors saved and independently reloaded; inner loss traces/IDs saved. Inner probe preprocessors themselves were discarded. |
| Core sequence expert | R9 sequence `sequence_mean` | Two fixed seeds averaged equally. Static/token transforms and last-training-year stopping use only the outer training population. Target-time tokens use only history at T−21 or earlier. | All fold/final weights and preprocessors saved; independent artifact audit also rebuilt sequence features and checked the stopping path. |
| Deferred different mechanism | R8 joint points `joint_surface` | Latent serve/return/surface parameters are re-estimated at each query cutoff using observations no later than T−21. A positive calibration slope uses earlier-season, previously generated as-of forecasts, with the outer 21-day purge. | Final snapshots and calibration metadata saved; audit checks calibration IDs/slopes and source chronology. Not every old snapshot was independently refitted. |
| Deferred sixth expert | R5 `both_cat_d4_it1200` | Fixed depth/iteration recipe; mirrored training uses only eligible pre-Y rows, with no outer-year early stopping or extra calibration. | Final model saved, historical fold estimators not saved. Its addition would need a separate mechanism justification before scoring, not a favorable 2023 row. |

The linear name matters. R5 `all_inputs_linear_C0.01` also has all 6,130 honest
per-year cached forecasts, but its final model was **not retained** by R5's
family selector. Using that precise alternative would require a new final
pre-2023 refit and should not silently substitute a different linear learner.
`r5_linear_logistic_C0.003` is the available, unambiguous native-logistic
primitive recommended for a cache-only pilot. This is an availability choice,
not a comparison of their 2023 accuracy.

The optional point expert is **prequential**, rather than fitted once before
each year: it can use earlier observations from the query's own season when
they are already at least 21 days old. This is allowed under the existing
information condition. It must not be described as having every latent weight
fit strictly before Y. Static classifiers also receive causal history features
that may contain earlier same-season records. The invariant is that the query
outcome and later records cannot enter its forecast.

The inspected generation paths exclude each predicted row's outcome from its
coefficients, stopping epochs and calibration. For R4/R5 this is code-and-cache
evidence, not an independent reload of missing historical estimators. The
claim is also conditional on the **fixed historical recipe**. R4/R5 recipes were previously selected during
adaptive development using these years, and all rounds have repeatedly viewed
2023. Their forecasts are not nested with respect to the entire research/model
selection process; this pilot cannot restore pristine confirmation.

## Inputs that must be rejected or unpacked

- R5 `accuracy_grid_blend`, and aliases such as `round5_pooled_comparator`, use
  weights selected from **all 2020–2022 OOF labels**. Reconstructing that same
  weighted sum for 2020 or 2021 does not make it independent of those labels.
  Do not feed it to a meta learner on these years. The early blend similarly
  uses 2020–2021 labels, although its 2022 evaluation was later in time.
- A global mixture or calibration fitted once on all 6,130 rows is a final
  meta model for later prediction, not an OOF feature or honest meta score on
  those same 6,130 rows. A fitted-on-own-label prediction from the final
  2009–2022 expert is also not a replacement for historical OOF forecasts.
- R9's individually prespecified fixed-weight blends do not themselves learn
  weights from labels, but the selected blend alias was chosen using OOF
  labels and is redundant with its primitive inputs. Use the raw seed average
  and R4 primitive, not a selected blend or a best-performing seed.
- R5 hinge probabilities are not automatically invalid: their calibration is
  inside the pre-Y training period, on earlier-trained margins from the last
  training year. However, they are outside the proposed small list. Do not
  confuse that valid inner calibration with the invalid pooled-OOF mixture.
- Names such as `previous_exploratory_best_loss` are selected comparators,
  not permission to add another expert. No expert list, context or subset may
  be chosen by a new 2023 leaderboard or error-overlap calculation.

R4's current recommendation is, in this case, the single named
`add_age_spline_C0.01`, not a learned mixture. Its explicit primitive column is
therefore usable under the recipe-selection caveat above. Checking that fact
is necessary; a generic field named “recommendation” is not enough.

## Reconstructed base-training chronology

For static fitted experts the original rule is
`season < Y and date <= min(date in Y) - 21 days`. The following values were
recomputed directly from the pinned 36,421 development targets:

| Predict Y | Query rows | First query date | Latest permitted fit date | Actual latest fit date | Fit rows / seasons |
| --- | ---: | --- | --- | --- | --- |
| 2020 | 1,214 | 2020-01-06 | 2019-12-16 | 2019-10-28 | 27,703 / 2009–2019 |
| 2021 | 2,388 | 2021-01-04 | 2020-12-14 | 2020-11-09 | 28,917 / 2009–2020 |
| 2022 | 2,528 | 2022-01-03 | 2021-12-13 | 2021-11-08 | 31,305 / 2009–2021 |
| 2023 | 2,588 | 2023-01-02 | 2022-12-12 | 2022-10-31 | 33,833 / 2009–2022 |

The neural/sequence inner tails are respectively 2019, 2020, 2021 and 2022;
their earlier fit counts are 25,201, 27,703, 28,917 and 31,305, with tail counts
2,502, 1,214, 2,388 and 2,528. Inner cutoffs are 2018-12-10, 2019-12-16,
2020-12-14 and 2021-12-13. The R9 audits independently reconstruct these masks.

The joint-point calibrators for 2020/2021/2022 use 7,550/8,764/11,152 forecasts
from seasons beginning in 2017, ending respectively on 2019-10-28, 2020-11-09
and 2021-11-08. Their purge cutoffs match the table. The corresponding saved
`joint_surface` slopes are 0.6124588485926197, 0.617639175621353 and
0.6256267272967736. These are distinct forward fits, not a single slope learned
using all outer-fold outcomes. Snapshot diagnostics additionally record newest
observations; each must be at or before its own cutoff.

## Minimal rigorous meta protocol, requiring no new base OOF fits

| Meta stage | Meta fitting rows | Honest query rows | Role |
| --- | --- | --- | --- |
| First forward comparison | 2020 primitive OOF, 1,214 rows | 2021, 2,388 rows | Select among the predeclared equal/global/conditional procedures, with the fixed accuracy and log-loss guard. |
| Practical gate | 2020–2021 primitive OOF, 3,602 rows | 2022, 2,528 rows | Refit the already chosen procedure; evaluate the existing +0.005 accuracy / +0.005 loss guard against the fixed R4 reference. |
| Final exploratory prediction | 2020–2022 primitive OOF, 6,130 rows | Full 2023, 2,588 rows | Refit the fixed procedure; freeze all meta parameters before computing its new 2023 predictions. Retain R4 as the branch recommendation if the gate fails. |

Apply the same 21-day date purge at every meta split, not only a season check.
For these actual rows the meta cutoffs are 2020-12-14, 2021-12-13 and
2022-12-12 and remove zero otherwise-eligible rows. Record exact meta fit/query
IDs nevertheless. Each query expert must be the corresponding year-specific
primitive, never the final expert re-predicting an earlier cohort.

The narrowest proposal freezes the procedure after the **2021-only** comparison
and uses 2022 only as the practical acceptance gate. The 4,916 honest meta
predictions from 2021–2022 can be reported together, but should not silently
reselect a procedure unless that additional rule is declared in advance.
The 2020 rows are meta training data: there is **no 2020 learned-meta OOF test**.
Do not manufacture one with resubstitution, fill it with R4, or call the result
a 6,130-row meta OOF score. An equal mixture is label-free, but its 2020 score
does not repair the missing learned-meta test. A genuine 2020 meta test would
require generating older primitive forecasts, as the original idea notes.

Use the same fixed experts and training rows for all three controls. Fit a
global simplex mixture and a small regularized linear softmax gate with at
most five declared **even** context inputs. A defensible available list is
`log_rank_abs_gap`, log1p minimum career count, log1p minimum surface-career
count, minimum 364-day serve-point log exposure, and `format_best_of_five`.
The serve-point minimum is exactly `(log_sum - abs(log_diff))/2`, because the
stored components are sums/differences of log1p counts. Confirm those semantics
and A/B invariance; do not infer them only from a column suffix.

Fit context transforms only on each meta-training population. No outcome-derived
reliability column, player ID vocabulary, random split, seed choice or new
regularization grid is needed. Freeze one shrinkage level and the optimization
rule in the eventual protocol. Center contexts on training rows and make a
zero-context conditional gate equal its global control. If global weights are
floored before taking logs, the same floored/renormalized weights must define
that control; otherwise “constant context reduces to global” is false.

At minimum, compare conditional versus learned global on both later years,
report training-only versus forward loss distinctly, and expose weight
variation. A win over only the equal mixture or R4 does not establish useful
conditional routing. Collapse to constant weights, unstable year-to-year
weights or an in-sample-only gain are valid negative results. A gate does not
create missing information and is not promised to reach 70%.

## Evidence bindings and limits

The inspected OOF SHA256 values are:

| File under `results/` | SHA256 |
| --- | --- |
| `round4/oof_predictions.csv` | `d8d6b70c777bd52f9d15ebdf39cbae1b9c4b10786702cab232eadff7f400d3b6` |
| `round5/oof_predictions.csv` | `48ac256c83ad0d479af95829b7db5054449744e0c263ea12ddc2a421a875724a` |
| `round9/neural/oof_predictions.csv` | `cfbdce14016db99add6f6011938e2e89c893313d5c04935387c96821f898eb0d` |
| `round9/sequence/oof_predictions.csv` | `4d013076dd10dd926b369e859185141e59b656f19a4d077f05604c224d2a0cb9` |
| `round8/joint_points/oof_predictions.csv` | `028e917213675a0357620aacfaa9ad56685f2b47de90344dd4b178da0cbf4171` |

Implementation paths inspected: `train_round4.py`, `round2_learning.py`,
`train_round5.py`, `round5_learning.py`, `train_models.py`, `tabular_neural.py`,
`train_neural_round9.py`, `sequence_learning.py`, `train_sequence_round9.py`,
`joint_point_model.py`, `train_joint_points_round8.py`, and the branch audits.
No old OOF optimizer was replayed in this feasibility review. R4/R5's absence
of saved fold estimators is an evidence limitation; if stronger provenance is
required, refit only the fixed selected primitives and save their folds before
meta learning, without searching new parameters.

The proposal's distinction between frozen-expert Bernoulli stacking and the
original jointly trained mixture-of-experts paper is appropriate. Its separate
completion-certified-history idea remains source-dependent; cached UTC fields
do not justify shortening T−21. Both are exploratory proposals after extensive
development reuse, not independent confirmation or SOTA evidence.

## Follow-through review of the fixed implementation

The subsequent [exact gating protocol](round10_gating_protocol.md) adopts the
four core primitives, 2021-only selection, the 2022 gate with no pooled
reselection, and six meta optimizer runs. Its fixed shared global floor is
0.02 and its conditional penalty is `0.01 * sum(theta**2) / 2`; neither is a
grid. Deferred point/tree alternatives remain outside that menu.

The independent review of `conditional_gate.py` found and requested one
interface correction: equal-length probability/context tables could previously
have different row indices. The corrected fit and prediction paths reject that
case, and fitting also rejects a mismatched label-Series index. All 11 synthetic
tests pass. An additional independent central-difference probe found maximum
gradient error 1.82e−11; fitting fully A/B-complemented synthetic inputs changed
complemented predictions by at most 9.80e−13. Identical-expert and constant-context
controls reduce to their expected baselines. No empirical tennis fit occurred
during this review. Runner and saved-artifact chronology remain separate checks
to complete when those files exist.
