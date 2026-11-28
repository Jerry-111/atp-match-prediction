# Round 9: full-data supervised tabular neural networks

Protocol recorded 2026-11-27, before new empirical scores. This is the R9-A
proposal from the [independent exploration record](round9_ideas.md), not a
performance claim. Earlier rounds and coordinator prompts were visible.

## Hypothesis, evidence and falsification

**Idea:** supervised fitting on all 33,833 training targets may recover useful
nonlinear interactions from the existing 326 columns that R7's frozen,
context-limited TabICL could not learn. **Rival explanation:** a flexible network
may mainly fit old-season noise. A matched-width ordinary MLP tests whether the
specific pre-tuned recipe adds value. A null or worse result is useful evidence;
no literature headline implies 70% on this chronological tennis population.

Primary methods checked on 2026-11-27:

- Holzmüller, Grinsztajn and Steinwart, NeurIPS 2024,
  [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf),
  especially the TD-S simplification and Appendix implementation details.
  The broad tabular results motivate a small experiment; §2.2 uses random
  train/validation/test splits, not our forward-time setting.
- [Official standalone code](https://github.com/dholzmueller/realmlp-td-s_standalone/tree/a8c73f75dbeae4ab1ba5bc92372c444ffa0f766e),
  pinned commit `a8c73f75dbeae4ab1ba5bc92372c444ffa0f766e`.
  `mlp.py` and `preprocessing.py` were inspected in full. This repository is
  **MIT licensed**; its complete notice is preserved in our adapted module and
  a verbatim source snapshot under the branch results. PyTabKit's separate
  license must not be substituted for this license.
- Contrary evidence: Grinsztajn, Oyallon and Varoquaux,
  [NeurIPS 2022 primary paper](https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf)
  describes the strong performance of trees on medium-sized tabular tasks and
  neural difficulties with irrelevant features and irregular target functions.
  This does not establish failure here, but counters an automatic neural-win assumption.

## Frozen two-recipe menu

Both use all original **326 R5 inputs**, no current prices, no new histories,
the same eligible Tour rows, uniform original-row weights, 3 hidden layers of
width 256, batch size 256, fixed seeds **1741, 1742, 1743**, and equal averaging
of all three seeds. No seed is selected or discarded. The other R9 sequence
branch uses recency weighting; that cross-branch difference is not a matched
architecture comparison.

| Name | Frozen recipe |
| --- | --- |
| `realmlp_td_s` | Author's numeric robust median/IQR scaling with min–max fallback and smooth clipping `z/sqrt(1+(z/3)^2)`; trainable input diagonal; NTP linear layers (standard-normal weights/biases, weight divided by sqrt input width), SELU, zero final layer; two-class cross-entropy with 0.1 label smoothing; Adam betas (0.9,0.95); base LR 0.04 with author's `0.5−0.5 cos(2π log2(1+15t))` schedule and scale/weight/bias LR multipliers 6/1/0.1 |
| `plain_mlp` | Identical input preprocessing and hidden widths; ordinary PyTorch Linear/ReLU layers with default initialization; two-class hard-label cross-entropy; Adam LR 0.001, betas (0.9,0.999), constant LR; no dropout or weight decay |

This is a **bundled recipe comparison**, not an isolated activation or optimizer
ablation. The network sizes are 216,136 and 215,810 parameters. TD-S is not the
full RealMLP-TD system: no numerical embeddings or full-library hyperparameter
machinery are claimed. All numeric columns already have missingness encoding;
categorical encoding from the standalone library is unnecessary here.

## Time, stopping, mirroring and refit

- Outer folds predict 2020, 2021 and 2022 using strictly earlier seasons and the
  existing inclusive 21-day purge relative to the first outer target date.
  Final fitting uses all **33,833** eligible original targets through 2022.
- Within each outer training population, the **last available season** is an
  inner validation tail. Inner fitting uses earlier seasons whose dates are
  at most the first tail date minus 21 days. Only this inner tail chooses epochs.
  Outer-fold labels, 2023 and final-test labels cannot select an epoch.
- For each seed, run a fixed **256-epoch maximum** and select the last epoch
  attaining the lowest inner **symmetric log loss**. Then discard the probe
  weights and refit from the same seed on the entire eligible outer training
  population for that many epochs. This includes the former inner tail.
  Keep the original 256-epoch LR schedule horizon in the refit: stop its prefix,
  rather than rescaling the learning-rate curve to the chosen duration.
  Epoch transport from an earlier subset to the full population is an assumption.
- Split before mirroring. Each original row is paired with negated `*_diff`
  inputs and complemented label; even columns stay unchanged. Fit preprocessing
  only on this mirrored training set. Each epoch shuffles all mirrored rows
  and uses complete 256-row batches, matching the author's `drop_last=True`;
  the incomplete random tail changes each epoch. No target is removed from
  prediction or scoring.
- Prediction averages `p(x)` and `1−p(swap(x))` for each seed, then all seeds.
  This enforces complementation and keeps threshold **0.5**. No probability
  temperature or threshold is estimated afterward. Seed variation is reported
  diagnostically, never used for selection.

Explicit adaptations from the upstream standalone code: chronological inner
tail plus full refit, symmetric log-loss stopping instead of classification-error
stopping, mirroring, fixed seed averaging, finite/date/schema guards, full-device
training tensors and a seeded permutation loop instead of DataLoader shuffling,
and frozen CPU state serialization. The architecture, TD-S loss, initialization,
batch size, optimizer groups and schedule formula are preserved. Numeric-only
preprocessing avoids the upstream categorical encoder's obsolete NumPy aliases.

## Fixed selection and audit

Two ensemble configurations, each with three seeds, are fitted in three outer
folds and once for 2023: **24 final network fits plus 24 inner stopping probes**.
They are two configurations, not 48 independent hyperparameter choices.
Add six deterministic R4 probability mixtures, new-model weights **0.25, 0.5,
0.75**. With the R4 reference there are **9 selectable entries**. The previous
R5 accuracy and probability comparators are report-only; all 11 probabilities
are retained. Seeds are not selection entries.

Choose accuracy first on 2020–2021, among entries whose log loss is at most
R4 +0.005; break accuracy ties by lower loss, then name. The chosen early entry
must gain at least **0.005 accuracy** with loss deterioration at most **0.005**
on 2022 to allow a branch-local upgrade. The pooled 2020–2022 choice is frozen
before new 2023 prediction regardless of that gate. This repeatedly used gate
is an exploration rule, not independent confirmation. Preserve the failed
choice if 2023 disappoints. Every 2023 result is exploratory and uses all
**2,588** original targets; **1,812** correct are needed for 70%. 2024–2025 stay sealed.

Verify source and feature hashes, reconstruct target metadata independently,
record exact training/inner-tail IDs, save all outer/final weights and inner
loss paths, and test original equations, date poisoning, split-before-mirror,
seed averaging, swapping, batch/permutation invariance and serialization.
Float32 inference may change by small rounding amounts across chunk shapes;
report measured errors rather than claiming exact arithmetic. The numerical
audit tolerance is 2e-6 probability and unchanged full-cohort classifications.

## Synthetic cost check and reproduction

The unscored synthetic benchmark used 8,192 original rows ×326, mirrored to
16,384, plus 1,024 synthetic inner rows for 8 epochs. CPU TD-S/plain took
0.832/0.510 seconds; MPS took 1.482/0.545 seconds. Values and weights were finite.
This is a timing check, not a real-data score. **CPU, four threads** is frozen
for the run and leaves MPS available to the sequence branch. The bounded run
is expected to take minutes to tens of minutes; no empirical hyperparameter
change is authorized by a poor result.

Use the already pinned isolated environment; do not install Torch into the
base modeling environment:

```bash
.venv-foundation/bin/python -m scripts.train_neural_round9
.venv-foundation/bin/python -m scripts.train_neural_round9 --predict-features results/round9/neural/inference_features.csv
.venv-foundation/bin/python -m scripts.train_neural_round9 --report-only
.venv-foundation/bin/python -m unittest tests.test_tabular_neural tests.test_train_neural_round9
```

The branch registry pins this protocol, local adapted code, upstream snapshot,
input hashes and the actual dependency snapshot before fitting. No new package
installation is needed. Results belong to the original historical T−21 input
condition, with no market-based training supervision.
