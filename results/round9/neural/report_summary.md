# Round 9: supervised tabular neural networks

Same 2,588 original targets and 0.5 decision threshold. These are repeated 2023 exploratory results; 2024–2025 remain sealed. No current-market inputs or market teacher labels are used.

Branch-local recommendation: **realmlp_td_s_r4blend0.75**. Early-selected 2022 gain is **+0.514 percentage points**, with log-loss change **-0.000133**; required gain +0.5 pp and allowed loss increase +0.005.

Pooled-CV frozen comparison `realmlp_td_s_r4blend0.75`: **1697/2,588 = 65.572%**, loss **0.607936**. It needs **115** more correct to reach 70%.

Two recipes each average the same three fixed seeds. All 33,833 original training targets and 326 columns are used. Each seed selects an epoch only on a purged inner training-period tail, then refits on all eligible training rows. There are 24 final outer/2023 network fits and 24 inner probes, two selectable fitted configurations and six deterministic R4 mixtures. R5 comparisons and individual seeds do not enter the nine-entry selector.

| Model | Correct / 2,588 | Accuracy | Log loss | Role |
| --- | ---: | ---: | ---: | --- |
| `round5_pooled_comparator` | 1715 | 66.267% | 0.609871 | report-only prior comparator |
| `plain_mlp_r4blend0.75` | 1709 | 66.036% | 0.607570 | fixed control / reference |
| `round5_probability_comparator` | 1706 | 65.920% | 0.606453 | report-only prior comparator |
| `plain_mlp` | 1706 | 65.920% | 0.608949 | fixed control / reference |
| `plain_mlp_r4blend0.5` | 1705 | 65.881% | 0.607238 | fixed control / reference |
| `realmlp_td_s` | 1705 | 65.881% | 0.608981 | fixed control / reference |
| `realmlp_td_s_r4blend0.5` | 1699 | 65.649% | 0.607670 | fixed control / reference |
| `plain_mlp_r4blend0.25` | 1697 | 65.572% | 0.607866 | fixed control / reference |
| `realmlp_td_s_r4blend0.75` | 1697 | 65.572% | 0.607936 | branch-local choice |
| `realmlp_td_s_r4blend0.25` | 1696 | 65.533% | 0.608161 | fixed control / reference |
| `round4_reference` | 1696 | 65.533% | 0.609407 | fixed control / reference |

## Matched recipe comparison

| Period | TD-S accuracy change vs plain / pp | TD-S loss reduction vs plain |
| --- | ---: | ---: |
| pooled | +0.196 | +0.000457 |
| 2020 | +1.153 | +0.003394 |
| 2021 | -0.544 | +0.001678 |
| 2022 | +0.435 | -0.002106 |
| 2023 | -0.039 | -0.000032 |

This isolates a bundled training recipe, not a single architecture component: TD-S changes initialization, activation, optimizer/schedule and label smoothing. Both have the same robust preprocessing, three 256-wide layers, uniform sample weights, seeds and chronology. It is an adaptation of the author’s standalone TD-S, not the full RealMLP-TD system or a replication of its random-split benchmark. Epoch transport to the full training population, historical drift and repeated development search limit interpretation. A posthoc highest row is not substituted for the frozen choice.

Run took 1796.1 seconds on CPU with four threads. Saved weights retain all seeds and folds; incomplete randomized training batches are omitted within each epoch as in the upstream recipe, while every prediction target remains in the cohort.

Reproduce: `.venv-foundation/bin/python -m scripts.train_neural_round9`. Reload: `.venv-foundation/bin/python -m scripts.train_neural_round9 --predict-features results/round9/neural/inference_features.csv`. Report only: `.venv-foundation/bin/python -m scripts.train_neural_round9 --report-only`.

[Protocol and evidence](../../../docs/research/round9_neural.md) · [Frozen selection](frozen_selection.json) · [Matched controls](matched_controls.csv) · [Inner stopping metadata](training_metadata.json) · [Synthetic timing](synthetic_smoke.json) · [Upstream MIT license](upstream/LICENSE.txt)
