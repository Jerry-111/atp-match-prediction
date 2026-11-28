# Round 7: privileged opening-market distillation

Inference uses only the existing T−21 historical features. Training additionally uses 2021–2022 opening-market supervision. No 2023 market prices are loaded. All numbers below describe the unchanged 2,588-match 2023 cohort and are exploratory; 2024–2025 remain sealed.

This market-supervised branch's recommendation: **market_lambda1**. Early selection: `market_lambda1`. The 2022 gate changes accuracy by +0.672 percentage points (required +0.5), loss by -0.003950.

Pooled H2+2022 choice, frozen before new 2023 scoring: **market_lambda1**.

| Model | Correct / 2,588 | Accuracy | Log loss |
| --- | ---: | ---: | ---: |
| uniform_lambda1 | 1716 | 66.306% | 0.622271 |
| uniform_lambda0.5 | 1715 | 66.267% | 0.608905 |
| previous_round5_exploration | 1715 | 66.267% | 0.609871 |
| market_lambda1 | 1712 | 66.151% | 0.604599 |
| uniform_lambda0.25 | 1709 | 66.036% | 0.606017 |
| hard_labels | 1706 | 65.920% | 0.606453 |
| previous_round5_best_loss | 1706 | 65.920% | 0.606453 |
| market_lambda0.5 | 1704 | 65.842% | 0.605159 |
| market_lambda0.25 | 1702 | 65.765% | 0.605703 |
| previous_probability_recommendation | 1696 | 65.533% | 0.609407 |

70% requires **1812 correct**. The best observed row above is `uniform_lambda1` with 1716 correct, still 96 short. This observed ordering does not alter selection.

All seven students fit 33,833 historical targets. 4,854 have eligible teacher probabilities; missing-price rows keep hard labels. Market and uniform teachers modify exactly the same rows. The uniform teacher is a required rival explanation, not a separate information improvement. The purely historical-label R4 recommendation remains a distinct condition.

| Lambda | Pooled CV market − hard accuracy (pp) | Market − uniform accuracy (pp) | Market − uniform loss |
| ---: | ---: | ---: | ---: |
| 0.25 | +0.087 | +0.262 | -0.000908 |
| 0.5 | +0.204 | +0.058 | -0.002684 |
| 1.0 | +0.494 | +0.640 | -0.008401 |

Against the identical hard-label learner, the frozen choice changes correct decisions by +6 and reduces log loss by 0.001855; its conditional 95% loss-reduction interval is [-0.000299, 0.003986]. This matched comparison is more informative about supervision than comparison with a different R4 architecture. An interval spanning zero does not establish a reproducible improvement.

The highest observed accuracy belongs to an unselected row and must be assessed together with its loss. In particular, heavy uniform smoothing can raise accuracy while degrading the probability estimates.

The full matched table includes both forward folds and exploratory 2023. Positive accuracy differences favor market supervision; negative loss differences favor it. Paired date-block intervals in summary.json are conditional descriptive intervals and do not account for the full adaptive search, repeated 2022 gate, or all player dependence.

Quotes are described as prematch by the provider, but capture timestamps are absent. Their recorded match times bound training eligibility under that source claim. The teacher may contain late fitness/news information unavailable to the student and impossible to reconstruct from T−21 history. This experiment cannot establish SOTA or validate a 70% claim.

Reproduce: `.venv/bin/python -m scripts.train_distillation_round7`. Reload without fitting: append `--predict-features results/round5/development_features.csv`. Only use trusted local joblib artifacts.
