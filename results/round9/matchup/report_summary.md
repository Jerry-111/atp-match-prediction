# Round 9: fixed antisymmetric style matchups

This tests new interactions in the existing T−21 information, not a new data
source. All 2,588 2023 targets and the 0.5 threshold remain unchanged. 2023 has
been repeatedly inspected and these results are exploratory; 2024–2025 stay sealed.

The branch-local rule **retain_round4**. Its early
choice `matchup_surface` changed 2022
accuracy by +0.0000%, versus the
required +0.5 percentage points and loss increase ≤0.005. The retained/adopted
choice `previous_probability_recommendation` scores 1696/2,588,
65.533%, log loss 0.609407.

The separately pooled pre-2023 choice **matchup_shared**
scores **1709/2,588 = 66.036%**, loss
**0.606459**. The highest observed row `previous_round5_exploration` scores
1715/2,588; it does not overwrite the frozen decision.
70% requires 1,812 correct. No threshold, subset or blend was selected on 2023.

## Full-cohort comparison

| model | correct | accuracy | mean_log_loss | recommended |
| --- | --- | --- | --- | --- |
| previous_round5_exploration | 1715 | 0.662674 | 0.609871 | False |
| previous_round7_market_student | 1712 | 0.661515 | 0.604599 | False |
| matchup_shared | 1709 | 0.660355 | 0.606459 | False |
| previous_round5_best_loss | 1706 | 0.659196 | 0.606453 | False |
| matchup_no_new_interaction | 1705 | 0.658810 | 0.606385 | False |
| matchup_surface | 1703 | 0.658037 | 0.606447 | False |
| previous_probability_recommendation | 1696 | 0.655332 | 0.609407 | True |
| rank_reference | 1640 | 0.633694 | 0.629796 | False |

## Matched structural tests

Positive accuracy change and negative loss change favor the candidate.

| candidate | control | oof_accuracy_change | oof_loss_change | validation_accuracy_change | validation_loss_change |
| --- | --- | --- | --- | --- | --- |
| matchup_shared | matchup_no_new_interaction | 0.000653 | 0.000070 | 0.001546 | 0.000074 |
| matchup_surface | matchup_no_new_interaction | 0.000163 | 0.000115 | -0.000773 | 0.000062 |
| matchup_surface | matchup_shared | -0.000489 | 0.000045 | -0.002318 | -0.000012 |

All three models use the same 188 old odd main-effect
columns, C=0.01 and three-year recency weighting. Shared adds 21 skew terms;
surface adds 63 strongly shrunk deviations. No learned player-ID vocabulary,
extra neural network, low-rank optimization or hyperparameter grid. The older
matrix already has a few handcrafted matchup terms; “no new interaction” does
not mean globally interaction-free. The 326 inputs are preserved exactly.

## Descriptive logit contributions

| model | design_width | main_mean_absolute_logit | shared_mean_absolute_logit | surface_mean_absolute_logit | coefficient_l2 |
| --- | --- | --- | --- | --- | --- |
| matchup_no_new_interaction | 188 | 0.834587 | 0.000000 | 0.000000 | 0.589974 |
| matchup_shared | 209 | 0.836021 | 0.016389 | 0.000000 | 0.605645 |
| matchup_surface | 272 | 0.835677 | 0.015165 | 0.004997 | 0.607821 |

These magnitudes describe fitted predictions, not causal effects of style.
The observed-style skew matrix has at most rank six by its dimension, without
an imposed low-rank factorization. Surface and shared columns partly overlap;
ridge yields a unique, shrunk decomposition. All centers/scales are fitted
only on each training fold; factors 0.2/0.1 are applied after RMS normalization.

Matched date-block intervals and model coefficients are saved in summary.json
and coefficients.csv. They condition on the fitted predictions and ignore
repeated search and some shared-player dependence. The primary Chen–Joachims
tennis experiment had no accuracy gain, so a null result here is meaningful;
this observed-feature adaptation is not a replication of that latent model.
R7's market-supervised student has a distinct training-information condition.

Reproduce: `.venv/bin/python -m scripts.train_matchup_round9`.
Inference only: append `--predict-features path/to/historical_features.csv`.
[Protocol](../../../docs/research/round9_matchup.md) ·
[Frozen choice](frozen_selection.json) · [Source registry](search_registry.json).
