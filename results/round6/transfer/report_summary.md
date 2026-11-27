# Round 6: supervised lower-level transfer

All results use the same 2,588 Tour matches in 2023 with T−21 historical information. They are exploratory after earlier inspection of 2023; 2024–2025 remain sealed.

Formal recommendation: **round4_reference**. Early-selected 2022 gain 0.158 percentage points (required 0.5), loss change -0.001434.

| Frozen or matched comparator | Correct / 2,588 | Accuracy | Log loss |
| --- | ---: | ---: | ---: |
| round4_reference | 1696 | 65.533% | 0.609407 |
| round5_pooled_comparator | 1715 | 66.267% | 0.609871 |
| transfer_linear_C0.1_aux0_r4blend0.75 | 1702 | 65.765% | 0.609280 |
| transfer_linear_C0.01_aux0 | 1700 | 65.688% | 0.612170 |
| transfer_linear_C0.1_aux0 | 1699 | 65.649% | 0.612269 |
| transfer_spline_C0.01_aux0 | 1698 | 65.611% | 0.612035 |
| transfer_linear_C0.1_aux0.1 | 1702 | 65.765% | 0.611093 |

Pooled 2020–2022 selection was frozen as `transfer_linear_C0.1_aux0_r4blend0.75` before scoring new 2023 predictions. Every raw and blended candidate is reported in validation_comparison.csv; observed 2023 ordering does not alter selection.

75261 additional supervision targets are available. Positive-auxiliary-weight models use 109094 eligible fitting rows, including 33833 Tour rows. The frozen pooled choice has auxiliary weight zero and only 33833 active Tour training rows. Auxiliary weight zero removes auxiliary rows before scaling. All weight sums normalize to the Tour fit count.

The matched CV table isolates auxiliary supervision with the same compact inputs and regularization. All 9 positive-auxiliary configurations worsen pooled CV log loss relative to their matched Tour-only control; only the linear C=0.1, auxiliary-weight=0.1 setting improves pooled accuracy. The rich Round 4/Round 5 references also contain technical/player information absent from this compact experiment. A null transfer result cannot establish that all future transfer approaches fail.

The 2022 gate and historical reference have been reused across rounds; this is a practical exploration rule, not independent confirmation. Challenger and qualifying labels may follow different conditional relationships. No SOTA or 70% claim is established by this experiment.

Reproduce: `.venv/bin/python -m scripts.train_transfer_round6`. Reload without fitting: `.venv/bin/python -m scripts.train_transfer_round6 --predict-features results/round6/transfer/inference_features.csv`.
