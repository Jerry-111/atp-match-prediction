# Round 10: conditional reliability of four frozen experts

Original historical T−21 condition; all 2,588 targets, labels and threshold 0.5 remain fixed. 2023 is repeatedly used exploratory validation. The 2024–2025 final test remains sealed.

The procedure frozen on 2021 alone is **round4_reference**. Its 2022 gate gains **+0/2,528** (+0.000 pp), with loss change +0.000000. Branch recommendation: **round4_reference**.

The unchanged selected procedure scores **1696/2,588 = 65.533%**, log loss **0.609407** on 2023; it is **116** correct short of 70%.

2020 supplies 1,214 meta-training observations, not a learned-meta test. The honest forward meta comparison contains 4,916 rows in 2021–2022; it is not used for pooled reselection. Final meta fitting uses 6,130 original primitive OOF forecasts through 2022. No base network or classifier was retrained.

| Procedure | Correct / 2,588 | Accuracy | Log loss | Role |
| --- | ---: | ---: | ---: | --- |
| `equal_mix` | 1687 | 65.185% | 0.606512 | fixed control/reference |
| `global_mix` | 1684 | 65.070% | 0.607014 | fixed control/reference |
| `conditional_mix` | 1684 | 65.070% | 0.607024 | fixed control/reference |
| `round4_reference` | 1696 | 65.533% | 0.609407 | 2021-selected procedure; branch recommendation |

## Does context help over the same global mixture?

| Period | Conditional accuracy change / pp | Conditional loss reduction | Mean probability change |
| --- | ---: | ---: | ---: |
| 2021 | +0.126 | +0.000152 | 0.001821 |
| 2022 | -0.079 | +0.000160 | 0.001029 |
| 2023 | +0.000 | -0.000010 | 0.000427 |

Equal/global/conditional procedures share the same four primitive forecasts. The global weights have a fixed 0.02 floor; conditional routing uses exactly that anchor and a fixed 0.01 quadratic penalty on 15 identifiable coefficients. Its five even contexts use only preknown rank separation, coverage and format. A win over equal averaging alone is not proof of useful routing. Training losses are optimizer diagnostics, not forward scores. No best-2023 row replaces the frozen choice.

R4/R5 old fold estimators were not stored: their primitive chronology relies on frozen masks/source/cache audits. R9 fold models were independently reloaded. Reused earlier model selection and the repeated 2022 gate limit confirmation. Neither a mixture nor this audit restores an untouched historical test.

Six optimizer runs completed in 0.49s. The numerical routine saves all three stages, context scalers, exact training IDs and weight spread. Probability clipping counts are explicit.

Reproduce: `.venv/bin/python -m scripts.train_gating_round10`. Reload: `.venv/bin/python -m scripts.train_gating_round10 --predict-inputs results/round10/gating/inference_inputs.csv`. Report only: `.venv/bin/python -m scripts.train_gating_round10 --report-only`.

[Protocol](../../../docs/research/round10_gating_protocol.md) · [OOF source legality](../../../docs/research/round10_oof_feasibility.md) · [Frozen choice](frozen_selection.json) · [Routing contrasts](matched_routing.csv) · [Weight spread](weight_spread.csv) · [Exact meta chronology](training_metadata.json)
