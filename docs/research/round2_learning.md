# Round-two learning proposal: smooth antisymmetric probabilities

Date: 2026-11-24. Contributor: audit/learning agent, AI-assisted independent
generation followed by implementation and method-documentation checks. Decision
owner: project user; the parent agent orchestrates the shared forward comparison.
This proposal uses the scientific-brainstorming workflow. The first-round 2023
results have already been seen, so this second-round search is exploratory even
though its tuning is restricted to 2020–2022. The untouched 2024–2025 holdout is
still required for final evidence.

**Focal question:** can smooth, regularized nonlinear response and context-dependent
reliability improve historical-strength probability estimates beyond the existing
linear, CatBoost and histogram-boosting learners?

## Independent candidates and decision

| ID | Idea | Assumption and predicted observation | Main falsifier | Disposition |
|---|---|---|---|---|
| L1 | Ridge logistic head with odd spline corrections and a small set of reliability interactions | Strength effects saturate smoothly; surface/history depth changes reliability. Forward-fold log loss could improve without the variance of fine tree partitions. | No stable forward-fold gain over the original linear model, or gain limited to one season. | Implement a bounded six-candidate experiment. |
| L2 | Residual boosting around a fixed rating logit offset | Trees can learn useful residual structure. | Offset calibration errors dominate; little incremental value over the existing boosted learners. | Defer because it overlaps the first-round tree family and complicates calibration. |
| L3 | Flexible learned calibration on recent seasons | Temporal drift changes probability sharpness. | Too few held-out calibration matches; slope changes fit noise. | Defer a separate calibrator; include a fixed three-year recency-weight option in L1. |

Criteria fixed before the new scores: chronological leakage safeguards and exact
player-exchange symmetry are gates; pooled 2020–2022 log loss is the primary
criterion; fold stability, complexity and runtime explain the choice. No novelty
claim or claimed empirical gain follows from these design arguments. A numerical
idea vote is unnecessary because the comparison can be measured directly.

## Implemented family

`scripts/round2_learning.py` exposes `candidate_specs(columns)`, `make_model(spec)`,
`fit(X, y, seasons=None)` and one-dimensional `predict_proba(X)`.

The linear part uses all declared difference features. Nine named strength and
rate differences, when present, receive quadratic B-spline corrections. Each
basis is made odd by taking `(B(z) − B(−z))/2`. The argument is scaled by training
root mean square; knots are fixed at −3, −1.5, 0, 1.5 and 3 scaled units. Constant
extrapolation makes the spline correction saturate outside its supported region.
The linear component remains, with symmetric clipping at eight scaled units.

Fourteen prespecified interaction pairs combine an odd strength signal with an
even context: surface indicators, mean rank, and history or point-opportunity
counts. Context centering and scales use the fitted training rows only. Context
values are clipped to three training standard deviations. No outcome or player
identifier is used to choose knots or transformations.

The design is odd, and the logistic head has no intercept, which gives
`P(A,B) = 1 − P(B,A)` by construction. The nonlinear blocks are scaled to 0.3
times their training RMS-normalized values, so their coefficients receive a
stronger effective ridge penalty than linear terms. This is an antisymmetric
additive model with ridge shrinkage, not a classical GAM with a learned smoothness
penalty or a monotonicity guarantee.

The six specifications combine `C ∈ {0.003, 0.01}` with three structures:
splines alone; splines plus interactions; or splines plus interactions and a
three-season exponential training-weight half-life. The caller must train each
candidate separately on each expanding fold. No internal cross-validation,
2023 scoring, calibration fitting or model selection occurs inside this module.

## Adversarial review and limits

The main alternative explanation for apparent gains is extra search flexibility
after inspecting first-round validation results. Another is an unusually favorable
season rather than stable response shape. All six candidates and fold scores must
remain visible, including negative results; the prior artifacts must not be
overwritten. The six candidates are a feasibility choice, not proof that the best
possible additive model has been searched.

Correlated rating and rate inputs can make individual coefficients unstable. The
model is a predictive instrument; spline coefficients are not causal effects.
Surface data remain sparse on grass. Constant tail extrapolation and stronger
ridge shrinkage mitigate variance without establishing good calibration. RMS
scaling is train-only but can still be affected by extreme training values.

The initial proposal was independently generated before this bounded check of
official method documentation. The documentation supports the B-spline API and
penalized logistic implementation; it does not establish tennis-specific benefit:
[SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)
and [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
checked 2026-11-24. No additional empirical literature search was conducted for
this learner; use the first-round evidence review for broader tennis context.

Tests cover exchange symmetry, reversing the training orientation, constant
inputs, finite extrapolation, frozen transforms, unused outcome poisoning,
recency-weight alignment and the bounded JSON registry. The accountable decision
will be made from the parent experiment's training-period comparison; test data
remain unopened.
