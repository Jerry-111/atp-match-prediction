# Round 5: classification objectives and restricted nonlinear residuals

Date: 2026-11-25, America/Los_Angeles. Owner: project user; implementation and initial hypotheses: Codex research agent. This is an AI-assisted exploratory proposal and implementation, not expert consensus or a claim that 70% accuracy is attainable. The scientific-brainstorming workflow and its reference material were used; procedural attribution is retained in [the evidence register](sota_evidence.md).

## Scope and independent initial proposals

Focal question: with the same complete 2,588-match 2023 cohort and T−21 historical information rule, can a different learning objective or restricted nonlinear representation improve classification accuracy beyond the existing probability-oriented models? No selective coverage, changed target labels, later historical results or 2024–2025 outcomes are allowed. The decision owner authorized accuracy-first follow-up, while retaining the previous log-loss recommendation and a probability-quality guard. Past 2023 results have already been viewed, so all new comparisons remain exploratory.

The following ideas were sent independently to the coordinating agent **before** reading other agents' proposals or conducting new literature searches. Available anchors were the existing project methods and earlier results. Criteria for prioritization were plausibility of genuinely different decision boundaries, cost, leakage resistance, interpretability of a null result, and preservation of all target matches; no numerical consensus score was used.

| ID | Initial idea and assumption | Disconfirming result | Cost and uncertainty |
| --- | --- | --- | --- |
| M1 | Select a small, fixed model/blend family by training-period accuracy rather than only log loss; their classification boundaries may differ. | Apparent early-fold gain does not persist in 2022, or probability loss worsens materially. | Low cost using existing OOF predictions. Small potential gain and substantial selection noise; not a credible standalone promise of four extra percentage points. |
| M2 | Compare logistic and squared-hinge objectives on identical antisymmetric designs; add a bounded random-feature residual as a distinct alternative to more trees. | No accuracy improvement at matched settings, deterioration across years, or strong dependence on arbitrary kernel choices. | Moderate compute; calibration requires a true temporal inner split. Ambiguous matches may contain little recoverable signal. |
| M3 | Let a few age/inactivity contexts modulate rank, Elo and point-strength differences. | Direction changes across years or pooled gains disappear outside a small subgroup. | Low-to-moderate cost. Existing age and dynamic-rating terms may already absorb the same information. |

Decision: the parent evaluates M1 and controls all model selection. This module implements M2 and a limited M3 block. New historical features from another agent enter the ordinary odd linear terms, but do not automatically expand the kernel input or interaction search. No separate human/domain expert took part in this ideation; independent agent review is requested rather than presented as validation.

## Bounded literature check and revision

Search date: 2026-11-25. Exact queries were `Bartlett Jordan McAuliffe 2006 convexity classification risk bounds classification calibrated hinge logistic loss pdf` and `Rahimi Recht 2007 Random Features Large Scale Kernel Machines NIPS pdf`. This was a two-method source check, not a comprehensive tennis literature review or novelty claim.

1. Bartlett, Jordan and McAuliffe (2006), *Convexity, Classification, and Risk Bounds*, JASA 101:138–156, DOI [10.1198/016214505000000907](https://doi.org/10.1198/016214505000000907). [Primary paper PDF](https://sites.stat.washington.edu/courses/stat527/s14/readings/Bartlett_etal_JASA_2006.pdf), introduction and Theorems 1–2. **Challenge located:** suitable logistic and margin surrogates can both be classification-calibrated; changing the loss does not inherently produce a better Bayes decision rule. The revised hypothesis is a finite-sample, model-class and regularization comparison, not an assertion that hinge is intrinsically superior.
2. Rahimi and Recht (2007), *Random Features for Large-Scale Kernel Machines*, NIPS 20. [Primary proceedings PDF](https://papers.neurips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf), §3, Eq. (2), Algorithm 1. **Method support located:** for the Gaussian kernel `exp(-gamma ||u-v||²)`, draw `omega ~ Normal(0, 2 gamma I)` and phases uniformly on `[0,2pi]`; the feature map is `sqrt(2/D) cos(omega.T u + phase)`. This is a scalable kernel approximation, not evidence of improved tennis accuracy.

Post-check refinement: preserve both objectives and explicit linear controls, limit the nonlinear expansion, and distinguish this project's antisymmetric projection and block weights from the original paper. No tennis-specific superiority claim follows from either source.

## Fixed candidate registry: 26 specifications

`scripts/round5_learning.py` exposes JSON-serializable `candidate_specs(columns)`, `make_model(spec)`, and new learners with `fit(x, y, seasons, dates=None)` / `predict_proba(x)`.

- Six representations: odd linear; odd linear plus context; and linear/context plus RFF with `D ∈ {256,512}` and width multipliers `{0.5,2}`.
- Each representation has logistic and squared-hinge heads with `C ∈ {0.003,0.03}`: 24 settings. Both use fixed three-year recency weights and zero intercept.
- Two unchanged existing antisymmetric spline controls use `C ∈ {0.003,0.01}` and a three-year half-life. These retain the existing `odd_spline` kind and fitting API; the new heads use `r5_logistic` / `r5_squared_hinge`.
- All stochastic maps use seed 158. No seed search, outcome-based feature filtering, early stopping against a scored outer fold, or expanding kernel-feature search is performed.

The kernel uses at most 20 explicitly named existing strength, serve/return, age, surface, format and inactivity fields. Its width is `gamma = multiplier / number_of_kernel_fields`. If a field is absent it is omitted by schema alone. All supplied `_diff` fields remain available to the linear part. Thus optional historical additions are accepted without making the nonlinear dimension depend on their count.

## Exact exchange symmetry and finite contexts

All raw `_diff` inputs change sign under A/B exchange; other fields remain fixed. Odd inputs are scaled by training RMS without centering. Even contexts use training means and scales. Eight possible interactions are fixed in source: rank×mean age, Elo×mean age, adjusted serve advantage×mean age, three age differences×mean log rank, inactivity difference×mean age, and age difference×total log inactivity. Unavailable pairs are omitted. They are all odd×even, preserving orientation.

Training-only RMS rescales each interaction; its block weight is 0.3. Odd inputs are clipped symmetrically to ±8 and centered contexts to ±3. For the RFF block, scaled kernel inputs are clipped to ±5. With `M` denoting player exchange, use the exact odd projection `psi(x) = [z(x)-z(Mx)]/2`. Multiplying by `sqrt(number_of_kernel_fields)` balances the block without increasing its expected norm merely because D doubles; a fixed residual block weight of 0.5 applies. These clipping, projection, context and block-scaling rules are **project adaptations**, not copied paper equations or tuned outcomes.

No-intercept linear heads over an odd design satisfy `f(Mx)=-f(x)`. Applying a positive scalar sigmoid yields complementary probabilities, and identical-player inputs receive 0.5. Arbitrary probability-threshold tuning is excluded: the decision threshold stays 0.5. Mirrored labels are unnecessary because the design already enforces the exact relation.

## Hinge probabilities require temporal calibration

Logistic heads use their native probabilities. For each squared-hinge fit:

1. The final year available **inside the outer training set** is the calibration year.
2. Fit a separate feature map and margin classifier on earlier years only. When dates are supplied, also require training dates ≤ earliest calibration date−21 days. With no dates, conservatively omit the whole immediately preceding year; this avoids pretending a year label identifies precise dates.
3. Predict margins for the held-out calibration year. Fit only a positive zero-intercept sigmoid slope, constrained to `[0.05,20]`, by Bernoulli loss on those out-of-time margins.
4. Refit the feature map and margin classifier on all outer training rows. Apply the frozen positive calibration slope to its future margins.

The calibration year is not included in the inner margin fit. Its labels can affect calibration and the final outer refit, but cannot affect the earlier-trained margins used to fit calibration. Outer validation labels never enter these steps. Training with seasons after 2022 is rejected. Metadata records exact inner sizes, years, purge cutoff and the calibration source.

The final refit's margin distribution can differ from the inner model's distribution. Transporting the slope is therefore an explicit approximation, not a guarantee of calibrated future probabilities. A positive slope preserves every classification decision; calibration cannot manufacture an accuracy gain. The coordinating runner's log-loss guard evaluates whether these probability outputs remain useful.

Independent agent review found no mathematical or real-data timing defect. It prompted an additional API guard: supplied fit dates must also be finite, aligned and before 2023, so relabeling a future date with an old season cannot bypass the training boundary. Exact date-year equality is intentionally not required: 188 legitimate fitting rows have the next season's designation while their tournament begins in the preceding December. This guard does not change the fixed candidate grid or any real-data fit.

## Tests, adversarial risks and handoff

`tests/test_round5_learning.py` uses synthetic pre-2023 years to check the 26-setting registry; matched objectives; unknown-feature handling; odd designs and predictions; unchanged training scales after arbitrary prediction inputs; zero at identical players; inclusive 21-day inner purge; whole-year fallback; later-label poisoning of the inner model; final refitting; positive-slope decision preservation; serialization; ignored extra labels; and invalid/final-holdout training rejection.

```sh
.venv/bin/python -m unittest tests.test_round5_learning -v
```

Remaining risks: choosing accuracy can overfit discrete changes in a few close matches; correlated features can dominate distance-based maps; RFF quality depends on a finite random basis; context trends may change over time; and calibration transport can fail even when classification improves. The fixed grid and separate temporal gate constrain these risks but do not turn repeatedly viewed 2022/2023 outcomes into fresh confirmatory evidence. Independent review should specifically challenge the inner calibration split, feature scaling, symmetry, and the parent's accuracy/log-loss gate.

No 2023 fit or scoring is performed by this implementation task. A single predetermined large RFF configuration may be timed on ≤2021 training rows and 2022 prediction inputs as an engineering benchmark; that timing will not be used to select feature definitions or hyperparameters. Model choice and all reported 2023 comparisons belong to the coordinating runner.

The fixed benchmark `r5_rff512_g2_squared_hinge_C0.03` fitted **31,305** rows through 2021 in **16.00 seconds** with four numerical-library threads; fitting plus 2,528 forward 2022 predictions and their exchanged counterparts took **16.05 seconds**. Predictions were finite and the maximum complementarity discrepancy was **2.22×10⁻¹⁶**. No accuracy or log loss was calculated. Inner calibration used 28,917 earlier fitting rows, the 2,388 rows of 2021 for calibration, and a 2020-12-14 training cutoff. These are runtime and split checks only; the fixed registry was not changed afterward.
