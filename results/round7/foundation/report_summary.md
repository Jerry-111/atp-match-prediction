# Round 7: frozen tabular foundation model

The complete 2,588-match 2023 Tour cohort and T−21 historical inputs are unchanged. These are exploratory results after repeated earlier inspection of 2023. Final 2024–2025 outcomes remain sealed.

The frozen branch choice scores **1686/2,588 = 65.147%**, log loss **0.614779**, on 2023. It does not improve the older Round 4 or Round 5 comparisons. The earlier Round 4 probability recommendation is preserved; this branch does not replace it globally.

Branch-local frozen accuracy choice: **tabicl_context4096_r4blend0.75**. The early-selected 2022 accuracy gain is **+1.028 percentage points** (required +0.5), with log-loss change -0.000134 (allowed at most +0.005).

The pooled 2020–2022 choice `tabicl_context4096_r4blend0.75` was frozen before new 2023 inference. Prior Round 5 comparators were excluded from every selection/gate. All candidates are retained below.

| Model | Correct / 2,588 | Accuracy | Log loss |
| --- | ---: | ---: | ---: |
| round5_pooled_comparator | 1715 | 66.267% | 0.609871 |
| round5_probability_comparator | 1706 | 65.920% | 0.606453 |
| round4_reference | 1696 | 65.533% | 0.609407 |
| tabicl_context4096_r4blend0.25 | 1696 | 65.533% | 0.609756 |
| tabicl_context4096_r4blend0.5 | 1696 | 65.533% | 0.611506 |
| tabicl_context2048_r4blend0.75 | 1696 | 65.533% | 0.614612 |
| tabicl_context2048_r4blend0.25 | 1695 | 65.495% | 0.609339 |
| tabicl_context2048_r4blend0.5 | 1690 | 65.301% | 0.611021 |
| logistic_context2048 | 1690 | 65.301% | 0.614260 |
| tabicl_context2048 | 1689 | 65.263% | 0.620428 |
| tabicl_context4096_r4blend0.75 | 1686 | 65.147% | 0.614779 |
| catboost_context4096 | 1686 | 65.147% | 0.622684 |
| tabicl_context4096 | 1682 | 64.992% | 0.619809 |
| catboost_context2048 | 1681 | 64.954% | 0.631195 |
| logistic_context4096 | 1676 | 64.760% | 0.613823 |

Two recent context sizes (2,048/4,096 original matches, doubled by player-swap mirroring) use identical 64-column inputs for TabICLv2, logistic regression and CatBoost. TabICLv2 uses one frozen synthetic-pretrained checkpoint, four ensemble members and no gradient fine-tuning. Conventional controls use fixed settings. Six deterministic TabICLv2/Round 4 mixtures add no fitted parameters.

Matched controls in matched_learner_controls.csv compare algorithms without changing rows or columns. Recent-context truncation can discard useful history; the older rich references use more training data and inputs. Generic tabular benchmark claims therefore do not determine the tennis result.

The source/dependency/checkpoint hashes, context IDs and unscored runtime/invariance smoke results are saved. MPS float32 kernels introduce small numerical differences across query chunks; the invariance tolerance is 2e−5, and serialized predictions are independently checked. The decision threshold stays 0.5.

The reused 2022 gate and conditional date-block intervals do not account for every earlier search. No final-test superiority or universal SOTA claim follows from this development run.

Reproduce: `.venv-foundation/bin/python -m scripts.train_foundation_round7`. Reload: `.venv-foundation/bin/python -m scripts.train_foundation_round7 --predict-features path/to/full337features.csv`. Refresh this report without fitting: `.venv-foundation/bin/python -m scripts.train_foundation_round7 --report-only`.

[Protocol and primary evidence](../../../docs/research/round7_foundation.md) · [Frozen selection](frozen_selection.json) · [Runtime smoke](smoke.json)
