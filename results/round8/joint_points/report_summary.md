# Round 8: jointly fitted point skills

The same 2,588 matches, original labels and 0.5 threshold are retained. All 2023 results are exploratory; 2024–2025 remain sealed.

Branch recommendation: **round4_reference**. Early-selected 2022 accuracy gain: **-0.198 percentage points** (requires +0.5), loss change +0.004546 (allows at most +0.005).

Pooled-CV frozen comparison `joint_recent_r4blend0.25`: **1689/2,588 = 65.263%**, log loss **0.609938**. It needs 123 additional correct predictions to reach 70%.

Five point-model structures jointly optimize all eligible past service observations at each T−21 cutoff. Fifteen fixed R4 mixtures add no separately fitted point models. Prior scales, history horizon and recency were fixed before fitting. Only earlier-season as-of predictions fit each positive calibration slope. Raw probabilities below are diagnostic, excluded from selection.

| Model | Correct / 2,588 | Accuracy | Log loss | Role |
| --- | ---: | ---: | ---: | --- |
| `joint_surface_event_r4blend0.5` | 1717 | 66.345% | 0.612987 | fixed control / prior comparator |
| `round5_pooled_comparator` | 1715 | 66.267% | 0.609871 | fixed control / prior comparator |
| `joint_surface_r4blend0.5` | 1713 | 66.190% | 0.613132 | fixed control / prior comparator |
| `round5_probability_comparator` | 1706 | 65.920% | 0.606453 | fixed control / prior comparator |
| `joint_recent_r4blend0.5` | 1701 | 65.726% | 0.613520 | fixed control / prior comparator |
| `serve_only_recent_r4blend0.5` | 1701 | 65.726% | 0.623906 | fixed control / prior comparator |
| `joint_static_r4blend0.25` | 1699 | 65.649% | 0.610338 | fixed control / prior comparator |
| `joint_static_r4blend0.5` | 1699 | 65.649% | 0.615064 | fixed control / prior comparator |
| `round4_reference` | 1696 | 65.533% | 0.609407 | branch choice |
| `joint_surface_event_r4blend0.25` | 1693 | 65.417% | 0.609743 | fixed control / prior comparator |
| `serve_only_recent_r4blend0.25` | 1693 | 65.417% | 0.612756 | fixed control / prior comparator |
| `joint_surface_r4blend0.25` | 1692 | 65.379% | 0.609811 | fixed control / prior comparator |
| `joint_recent_r4blend0.25` | 1689 | 65.263% | 0.609938 | pooled frozen comparison |
| `round2_adjusted_linear_comparator` | 1687 | 65.185% | 0.611117 | fixed control / prior comparator |
| `joint_recent_r4blend0.75` | 1687 | 65.185% | 0.619955 | fixed control / prior comparator |
| `joint_surface_r4blend0.75` | 1686 | 65.147% | 0.619167 | fixed control / prior comparator |
| `joint_surface_event_r4blend0.75` | 1685 | 65.108% | 0.618931 | fixed control / prior comparator |
| `joint_static_r4blend0.75` | 1678 | 64.838% | 0.623184 | fixed control / prior comparator |
| `joint_recent` | 1668 | 64.451% | 0.629343 | fixed control / prior comparator |
| `raw_joint_recent` | 1668 | 64.451% | 0.642211 | raw diagnostic only |
| `joint_surface_event` | 1665 | 64.335% | 0.627656 | fixed control / prior comparator |
| `raw_joint_surface_event` | 1665 | 64.335% | 0.647356 | raw diagnostic only |
| `joint_surface` | 1662 | 64.219% | 0.628013 | fixed control / prior comparator |
| `raw_joint_surface` | 1662 | 64.219% | 0.647872 | raw diagnostic only |
| `serve_only_recent_r4blend0.75` | 1652 | 63.833% | 0.641762 | fixed control / prior comparator |
| `joint_static` | 1642 | 63.447% | 0.634693 | fixed control / prior comparator |
| `raw_joint_static` | 1642 | 63.447% | 0.653704 | raw diagnostic only |
| `serve_only_recent` | 1564 | 60.433% | 0.666400 | fixed control / prior comparator |
| `raw_serve_only_recent` | 1564 | 60.433% | 0.675704 | raw diagnostic only |

## Matched structural controls

| Contrast | Pooled CV accuracy change / pp | Pooled CV loss reduction | 2023 accuracy change / pp | 2023 loss reduction |
| --- | ---: | ---: | ---: | ---: |
| opponent_adjustment | +4.633 | +0.041099 | +4.019 | +0.037056 |
| recency | +0.881 | +0.011969 | +1.005 | +0.005350 |
| surface_preference | +0.179 | +0.002005 | -0.232 | +0.001331 |
| event_intercept | -0.033 | -0.000036 | +0.116 | +0.000357 |

The old R2 adjusted-linear comparator has a different history/classifier/input system, so it is not a same-input method control. The nested structural comparisons above keep the history population and priors fixed except for the explicitly named term/weight. IID points, MAP plug-in uncertainty, standardized seven-point deciding-set tiebreaks, missing point records and reused validation remain limitations.

Source: 44,085 valid historical matches / 44,164 eligible inventory matches. Missing counts remove likelihood observations only, never targets. Each cutoff uses a four-year observation pool. Earlier 2023 points can update latent skills only once T−21 eligible; 2023 never estimates calibration or chooses settings. Event identity includes source code, normalized name and surface to avoid known reused-code collisions.

This is a sparse penalized-likelihood adaptation of Ingram (2019), not a claimed reproduction of the paper’s NUTS random-walk posterior. The published model itself was slightly worse than its Elo benchmark; no literature result proves 70% here.

Reproduce: `.venv/bin/python -m scripts.train_joint_points_round8`. Reload: `.venv/bin/python -m scripts.train_joint_points_round8 --predict-targets results/round8/joint_points/inference_targets.csv`. Report-only: `.venv/bin/python -m scripts.train_joint_points_round8 --report-only`.

[Protocol and primary evidence](../../../docs/research/round8_joint_points.md) · [Frozen choice](frozen_selection.json) · [Matched controls](matched_controls.csv) · [Synthetic timing](synthetic_smoke.json)
