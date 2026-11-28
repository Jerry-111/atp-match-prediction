# Round 9: a bounded antisymmetric style-matchup test

Recorded before any new empirical fit or score. This implements F9-1 from
[the independent idea register](round9_ideas.md). The features agent proposed
the mechanism; the audit agent implements it and another agent will review.
Scientific-brainstorming was used for the earlier exploration and this bounded
design, including negative evidence. These are AI collaborators, not a human
expert panel. Earlier 2023 results and the 70% goal are acknowledged anchors.

## Hypothesis and evidence

**Idea:** residual differences in player styles might form matchup effects that
a scalar ordering cannot express. **Prediction:** shared skew interactions
improve over identical main effects in forward years; surface deviations add
value only if these interactions vary with the surface. **Counterevidence:**
the added parameters might merely absorb nonlinear main effects, measurement
noise, or correlations already represented by existing handcrafted features.
Failure to improve the matched controls rejects practical value for this
bounded representation, not every possible form of nontransitivity.

Primary source checked directly on 2026-11-27: Chen and Joachims, WSDM 2016,
[*Modeling Intransitivity in Matchup and Comparison Data*](https://www.cs.cornell.edu/~tj/publications/chen_joachims_16a.pdf),
[DOI 10.1145/2835776.2835787](https://doi.org/10.1145/2835776.2835787).
Their Eq. 4 uses an antisymmetric difference of cross-player inner products
plus strength differences. Section 4.4 reports a small tennis likelihood gain
but no accuracy improvement. Their random splits do not establish temporal
generalization here. This source motivates a small falsifiable test rather
than a large search or a promise of 70%. Our observed-feature, convex ridge
adaptation does not reproduce learned player-ID embeddings or paper results.
This is a focused source verification, not a systematic review.

## Fixed information and transformation

Use the exact prior 36,421 × 326 historical feature matrix: 33,833 eligible
2009–2022 training rows and the original 2,588 2023 matches. No new source,
target, row removal, label correction, player vocabulary, or current quote.
The existing seven technical-rate sum/difference pairs already contain each
player's marginal rate; reconstruct `a=(sum+diff)/2`, `b=(sum-diff)/2` for ace,
double fault, first-in, first-won, second-won, return-first-won and
return-second-won. These inputs were computed with the existing T−21 lag,
364-day window and fixed historical priors. New information is not claimed.

Fit a common center and standard deviation to the concatenated A and B
training rates only. Reuse those parameters for both players and all later
queries; clip standardized rates to ±4. A near-zero standard deviation uses
scale 1. Form all 21 pairs `z_ij=a_i*b_j-a_j*b_i`, `i<j`. These change sign
under A/B exchange and are equivalent to `a^T K b` for a skew matrix K.
A 7×7 skew matrix has 21 free coefficients and rank at most six; no additional
low-rank constraint or low-rank optimization is claimed.

Existing 326 fields remain unchanged. The baseline uses their 188 odd
`*_diff` main-effect columns. It is called **no new interaction**, because the
old matrix already contains manually constructed first/second-serve matchup
and handedness terms. Some new directions can overlap these older columns;
this limits a pure causal/nontransitive interpretation. All three structures
have exactly the same baseline design, recency weights and regularization.

## Exactly three structures

1. **no_new_interaction:** only the old odd main-effect columns.
2. **shared:** append 21 shared skew-style terms.
3. **surface:** append the same 21 terms and their 63 interactions with the
   existing hard/clay/grass indicators. An all-zero indicator vector, including
   historical Carpet, uses shared effects only.

Main-effect columns use training RMS without clipping or centering, preserving
antisymmetry. Skew terms use their own training RMS (no mean subtraction).
Their standardized design is multiplied by **0.2**; surface deviations use the
same shared RMS and a factor **0.1**, without rescaling each sparse surface
subset. With a single ridge penalty this gives surface deviations four times
the coefficient penalty of shared interactions, partially pooling them toward
the shared effects. The strictly convex ridge objective identifies the
decomposition even when shared and surface columns overlap.

Every structure uses no-intercept logistic regression, **C=0.01**, three-year
recency half-life, seed 158, at most 2,000 optimizer iterations, and four CPU
threads. Minimize weighted Bernoulli negative log likelihood plus
`||beta||²/(2*C)`. One orientation is sufficient because the design is exactly
odd; fitting mirrored rows with half weights has the same objective.
No calibration, seed selection, C search, clipping search, blend or threshold
search. Three outer folds plus final fit give **12 training instances**.

The surface design is a small fixed extension; failure does not trigger a
larger rank/width/regularization menu. Synthetic-only tests and runtime checks
precede real fitting. All learned transforms must fit exclusively on the
current chronological training mask.

## Selection, evaluation and audit

Generate original 2020/2021/2022 expanding-year OOF predictions, with training
season earlier than validation and latest allowed date ≤ first validation
date−21 days. Only the three new structures and the R4 reference are eligible.
Select pooled 2020–2021 accuracy, breaking ties by lower log loss then name,
subject to loss ≤ R4 +0.005. Test that early choice on 2022; adoption requires
accuracy gain ≥0.005 and the same loss guard. Choose the pooled 2020–2022 entry
by the same rule and freeze it before 2023 scoring; the gate decides whether
the branch adopts it or retains R4. This repeatedly used gate is exploratory,
not independent confirmation. R5 and R7 comparators remain report-only.

Fit all three final models on the same eligible pre-2023 rows. Score all 2,588
matches at threshold 0.5; 70% requires at least 1,812 correct. Save every
configuration, OOF probability, 2023 probability, matched contrast, frozen
choice, fitted transform, model and source hash. Report coefficient norms,
shared/surface contribution magnitudes and paired date-block intervals as
descriptive evidence. They do not establish a causal player-style mechanism.

Tests cover hand-calculated skew terms; cyclic synthetic matchups that additive
ratings cannot represent; no interaction at identical styles; exact swap
complementation; train-only transforms and future-label poisoning; surface
partial pooling/scaling; column order; joblib reload; and chronological
selection. Independent audit reloads all probabilities and recomputes scores,
selection, gradients and transforms where feasible. No 2024–2025 data or final
test evaluation is permitted. Prior inspection makes all 2023 results
exploratory regardless of the pre-fit record here.

The coordinator reviewed and approved this exact three-structure menu before
empirical fitting. Proceed only after the synthetic implementation checks pass.
