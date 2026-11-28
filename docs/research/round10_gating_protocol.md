# Round 10: a small conditional mixture of four frozen experts

Prepared 2026-11-28, **before empirical meta fitting**. This is the finite
implementation plan for L10-A in [the independent idea record](round10_learning_ideas.md).
The coordinator fixed the four primitives by mechanism and saved-model
availability, not a new ordering of their 2023 errors. The
[independent OOF feasibility audit](round10_oof_feasibility.md) binds their
identities, chronology, source hashes and evidence limits. The previous nine
rounds and repeated 2023 results remain acknowledged research-selection anchors.
This protocol cannot make those years pristine confirmation.

## Question, cohort and fixed experts

Can preknown coverage/format context improve the reliability weights of the
**same four expert forecasts**, relative to a learned global mixture? Preserve
the original historical T−21 information condition, all 2,588 original 2023
targets, labels and threshold 0.5. The target for 70% is 1,812 correct.
2024–2025 remain sealed. No new source, target filtering or primitive retraining
is part of this experiment.

The following order is fixed throughout matrix construction and serialization:

| k | Primitive probability column | Frozen OOF file under `results/` |
| --- | --- | --- |
| 1 | `add_age_spline_C0.01` | `round4/oof_predictions.csv` |
| 2 | `r5_linear_logistic_C0.003` | `round5/oof_predictions.csv` |
| 3 | `realmlp_td_s` | `round9/neural/oof_predictions.csv` |
| 4 | `sequence_mean` | `round9/sequence/oof_predictions.csv` |

These are respectively an additive spline, native-logistic linear model,
three-seed tabular neural average and two-seed pooled-history encoder. Seeds
are not selectable experts. There are exactly 6,130 matched OOF rows:
1,214 / 2,388 / 2,528 in 2020 / 2021 / 2022. Source IDs, seasons and labels must
match the fixed development table, with no dropping or imputation. Final 2023
primitive predictions come from their frozen pre-2023 models and must have
the same 2,588 IDs. Any cache/hash/alignment failure stops the run.

No selected blend alias is a training input. R5's pooled weights learned on
all OOF labels, for example, would contaminate a new learner on those rows.
R4's actual recommendation is the **single first primitive**, as independently
checked; its explicit column is both a valid primitive conditional on the old
recipe and the reference probability. Do not assume other recommendation
aliases have the same property. R4/R5 did not retain old fold weights, so their
OOF chronology evidence is the frozen implementation, masks and prior audits;
R9's complete fold weights were independently replayed. This limitation is retained.

## Exactly three mixture procedures

Let `P[i,k]` be the four probabilities, clipped once to `[10^-6,1−10^-6]`
for **all three controls**. Report the number changed; preserve raw reference
scores separately. Training observations have uniform weight. No temperature,
intercept calibration, threshold fitting or further expert selection is added.

1. **Equal mixture:** `q_i = mean_k P[i,k]`. No fitted parameter.
2. **Learned global mixture:** minimize mean Bernoulli log loss
   `L(w) = mean_i[-y_i log(P_i w) − (1−y_i) log(1−P_i w)]`, subject to
   `sum(w)=1` and every `w_k >= 0.02`. These three free weights are the global
   control and the exact conditional anchor.
3. **Conditional mixture:** use the same global `w` fitted on the same rows,
   frozen during conditional optimization. For the five standardized contexts
   `z_i`, define `g_i = softmax(log(w) + z_i Θ Hᵀ)` and `q_i = sum_k g_ik P_ik`.
   `Θ` has shape 5×3, no new intercept. `H` has orthonormal columns
   `(1,−1,0,0)/sqrt(2)`, `(1,1,−2,0)/sqrt(6)` and
   `(1,1,1,−3)/sqrt(12)`. It removes the softmax common-logit ambiguity.
   Minimize
   `J(Θ) = mean_i BernoulliLoss(y_i,q_i) + (0.01/2) * sum_jk Θ_jk²`.
   The penalty is a **sum over 15 coefficients**, not a mean, and it is not
   divided by sample count again. It shrinks context dependence to zero.

The coordinator fixed the common global floor at **0.02 before empirical fitting**.
The initial implementation sketch proposed 10^-6; review identified that its
very negative log anchor could prevent a globally weak expert from becoming
useful in a particular context under the fixed penalty. This is an unscored
design correction, not a floor grid. The four floors reserve total weight 0.08;
the largest possible global weight is 0.94. Conditional weights are positive
softmax weights and may move above or below 0.02. They are not clipped to that
floor. The same floored global vector is always the comparator and anchor.

At `z=0`, conditional weights equal `w` mathematically. If all contexts are
constant in meta training, their transformed values stay zero at every later
query, `Θ=0` is optimal, and the conditional procedure reduces to that identical
global control. The implementation directly assigns `w` for zero-context rows
to avoid a needless softmax rounding difference. Convex combination and even
contexts preserve A/B complementation; they cannot create information absent
from the four experts and contexts.

## Five fixed, even contexts and train-only transformations

All contexts come from the unchanged 326-column R5 matrix, joined by target ID.

| Context | Exact original-field transform |
| --- | --- |
| Rank separation | `log_rank_abs_gap` |
| Overall experience | `log1p(career_min_count)` |
| Same-surface experience | `log1p(surface_career_min_count)` |
| Recent technical coverage | `log1p(points_overall_364_serve_points_min)` |
| Format | `format_best_of_five` |

All five are invariant under player exchange. The two career fields contain
raw minimum match counts (`advanced_features.py`, historical-count outputs).
`serve_features.py` stores `points_overall_364_serve_points_min` as the minimum
**raw service-point count**, so it receives exactly one `log1p`. It is
equivalent to `(serve_points_log_sum − abs(serve_points_log_diff))/2`; applying
`log1p` to that latter expression would be wrong and is not the chosen path.

Fit each context's population mean and population standard deviation only on
the current meta-training rows. If standard deviation is at most 10^-12,
mark the field inactive and always output zero, including at surprising future
values. Otherwise divide by `max(sd,10^-6)`, then clip to [−3,3]. Refit this
transform separately for the 2020, 2020–2021 and 2020–2022 fitting populations.
No full-OOF scaler, outcome-derived reliability field, player vocabulary or
additional surface/context split is permitted. Exact DataFrame index checks
prevent equal-length but shuffled probability/context/label pairings.

## Genuine forward meta chronology and procedure choice

For query year Y, use only OOF rows with season less than Y and tournament
date no later than the first query date minus 21 days, inclusive. Record exact
fit/query IDs and cutoffs even though this purge removes zero actual rows here.

| Stage | Fit meta weights/scaler on | Predict | Role |
| --- | --- | --- | --- |
| First forward comparison | 2020: 1,214 rows, cutoff 2020-12-14 | 2021: 2,388 rows | Select the procedure |
| Practical gate | 2020–2021: 3,602 rows, cutoff 2021-12-13 | 2022: 2,528 rows | Evaluate that already chosen procedure |
| Final exploratory comparison | 2020–2022: 6,130 rows, cutoff 2022-12-12 | 2023: all 2,588 rows | Refit the fixed procedure; no reselection |

The first learned-meta test is **2021**. There is no 2020 learned-meta OOF
prediction or score; no resubstitution, backfilled R4 probability or random
cross-validation will manufacture one. Report the 4,916 genuine 2021–2022
meta forecasts separately from the 6,130 meta-training rows available for the
final fit. Training loss is only an optimizer diagnostic, never forward accuracy.

Select on **2021 only**, from equal/global/conditional mixtures plus the R4
reference: **four entries**. An eligible entry has mean log loss no more than
reference +0.005. Choose highest accuracy, then lowest log loss, then lexical
name, using threshold 0.5 throughout. On 2022, the selected entry must improve
accuracy by at least 0.005 and increase loss by no more than 0.005 versus R4.
Otherwise the branch recommendation retains R4, while still reporting the
unsuccessful selected procedure. The repeatedly reused 2022 gate is practical
exploration, not independent confirmation. **There is no pooled-2021–2022
reselection.** Pooled metrics may be descriptive only.

Generate and retain all three 2023 controls once, plus the fixed reference.
The four report columns are exactly `equal_mix`, `global_mix`, `conditional_mix`
and `round4_reference`; no additional old comparator enters this branch's table.
The unaltered four primitive input columns are saved separately as provenance,
not as additional selectable procedures. Do not replace the 2021-selected procedure
with the best 2023 row. Conditional-versus-global is the matched routing contrast
on 2021, 2022 and 2023; beating only the equal mixture or R4 does not establish
that context-dependent routing helped.

## Optimization, checks and exact fit count

Global optimization uses float64 SciPy SLSQP, uniform initialization, analytic
gradient, maximum 1,000 iterations and `ftol=10^-12`. Bound/equality roundoff may
be repaired inside the same constrained simplex; verify objective change below
10^-8 and constrained KKT residual at most 10^-5. No unbounded global optimizer
is fitted as an extra control.

Conditional optimization uses float64 L-BFGS-B from the **single zero** Θ,
maximum 1,000 iterations, `ftol=10^-13`, `gtol=10^-9`, `maxls=50`, and requires
final maximum absolute gradient at most 10^-6. This nonlinear optimization is
not claimed to find a global optimum. Numerical failure must be recorded;
do not try multiple outcome-selected starts or quietly change the penalty.

With `d_i=(q_i−y_i)/(q_i(1−q_i))`, the conditional gradient is
`Zᵀ [d[:,None] * g * (P−q[:,None])] H / n + 0.01 Θ`.
For global weights it is `Pᵀ d / n`. Synthetic central finite differences,
zero-context identity, simplex-boundary optima, A/B inversion, constant and
identical experts, query batching/permutation, serialization, schema alignment,
fractional-label rejection and poisoned date/purge boundaries are required.
A constructed minority-context case must show that an expert at the global
0.02 floor can receive appreciably greater conditional weight; this verifies
capability, not empirical performance. Nearly identical forecasts should not
produce large learned routing.

There are **three algorithms**, only **two fitted meta procedures per stage**:
three global fits and three conditional fits, **six optimizer runs total**.
The conditional model reuses the already fitted global anchor; it does not
fit another hidden global control. Equal averaging fits nothing. There are
**zero new base-expert fits**, zero new seed choices and no additional mixture
weight grid. CPU threads are capped at four; the numerical problem is tiny.

Save source/input hashes, recipe ordering, all three stages' parameters and
context normalizers, optimization diagnostics, exact fit/query IDs, genuine
meta forecasts, selections, all complete-cohort 2023 controls and inference
reload. Report average/spread of weights and maximum probability changes by
stage, with no outcome-selected strata. Collapse to global weights, unstable
forward improvements, or in-sample-only gains are valid negative findings.
A loss gain without an accuracy gain is a probability result, not achievement
of the 70% goal. Keep descriptive intervals conditional on this extensive search.

## Primary basis and adaptation limits

[Jacobs et al. (1991)](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf),
pp.80–81 and Figure 1, motivate input-dependent normalized expert weights.
Their experts are jointly trained on a different task. This implementation
instead freezes expert forecasts and fits a Bernoulli mixture on chronological
held-out predictions; it is not presented as a reproduction.
[Van der Laan, Polley and Hubbard (2007)](https://pubmed.ncbi.nlm.nih.gov/17910531/)
motivate cross-validated combination weights. The verified original abstract
supports that principle, not a theorem guaranteeing this dependent, repeatedly
reused time series. [Kovalchik (2016)](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2015-0059/html),
Table 5, shows that tennis prediction difficulty differs across contexts; it
does not prove that conditional routing can exploit complementary errors here.
Our earlier failed global mixtures and the small 2020 warmup are counterevidence
against assuming a large gain. No paper establishes SOTA or 70% for this cohort.

Implementation: [conditional_gate.py](../../scripts/conditional_gate.py) and
[train_gating_round10.py](../../scripts/train_gating_round10.py).
Synthetic tests: [test_conditional_gate.py](../../tests/test_conditional_gate.py)
and [runner tests](../../tests/test_train_gating_round10.py).
Coordinator review approved this exact experiment before empirical fitting.
The runner writes input/source/protocol hashes first, then the 2021 choice,
then the 2022 gate, then all final meta parameters before loading 2023 primitive
probabilities for new mixtures. It does not refit any primitive.

```bash
.venv/bin/python -m pytest tests/test_conditional_gate.py tests/test_train_gating_round10.py
.venv/bin/python -m scripts.train_gating_round10
.venv/bin/python -m scripts.train_gating_round10 --predict-inputs results/round10/gating/inference_inputs.csv
.venv/bin/python -m scripts.train_gating_round10 --report-only
```

The default reload uses the final 2023 meta model. `--stage 2021` or `--stage 2022`
requires inputs for that year and uses its saved earlier-training meta model;
the final meta weights must never be substituted for those historical forecasts.
All three stages' global/conditional objects are saved in one inference bundle.
No dependency changes beyond the existing pinned base environment are required.
