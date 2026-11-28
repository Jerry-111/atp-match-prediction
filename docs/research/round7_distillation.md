# Round 7: market-supervised historical student

Protocol recorded on 2026-11-27 before fitting this branch. The same 2023 season
has already been inspected in rounds 1–6, so this is an exploratory experiment,
not a new confirmatory test. The 2024–2025 outcomes remain sealed.

## Hypothesis and rival explanation

**Idea A7-1 (independent, AI-assisted):** past opening-market probabilities can
provide less noisy supervision for a model whose inference inputs remain the
existing T−21 historical features. This changes the **training information**;
call it a market-supervised historical student, not a purely historical-label
algorithm. No market value is required or consumed at inference.

Lopez-Paz, Bottou, Schölkopf and Vapnik, *Unifying distillation and privileged
information*, ICLR 2016, equation (4), combines hard-label and teacher losses;
the teacher may observe information unavailable to the student at test time.
[Author's primary PDF](https://leon.bottou.org/publications/pdf/iclr-2016.pdf).
Our external bookmaker teacher is an adaptation, not a replication of a tennis
experiment in that paper. We fix teacher temperature at 1 and use normalized
two-sided opening implied probabilities.

**Rival explanation:** improvements may be ordinary soft-target regularization,
rather than useful market supervision. Yuan et al., *Revisiting Knowledge
Distillation via Label Smoothing Regularization*, CVPR 2020, relates these
mechanisms. [Primary paper](https://arxiv.org/abs/1909.11723). Therefore the
same rows, weights, model, lambda and availability mask are also evaluated with
a constant 0.5 teacher. Neither paper establishes 70% accuracy for this task.
This bounded source check followed independent idea generation; it is not an
exhaustive literature review. The broader skill provenance is recorded in the
round-seven research register.

## Fixed experiment

- Reuse the audited Round 5 historical matrix (326 available columns) and its
  `all_inputs_spline_C0.01` architecture: antisymmetric additive splines,
  C=0.01, three-year recency half-life. There is no new hyperparameter grid.
- Seven configurations: hard labels; market teacher at lambda 0.25, 0.5, 1;
  constant 0.5 teacher at those same three lambdas. The model's original odd
  feature selection and even-context interactions remain identical.
- Preserve all eligible Tour training rows from 2009 onward. Only rows with
  usable 2021–2022 opening prices have target
  `t=(1-lambda)*y+lambda*q`. Other rows keep the original binary `y`.
  The uniform-teacher control uses exactly the same available-price rows.
- Optimize weighted binary cross entropy by two pseudo-observations per
  original design row, with labels 0 and 1 and weights `w*(1-t)` and `w*t`.
  Their weight mass is exactly the original recency weight; C is unchanged.
  All transforms fit on original training rows only. The hard-label control
  uses the original classifier unchanged, and is checked against its old fit.
- Reuse **fixed outcome-blind Round 5 source links**, not a new match search.
  Read only identities from that linkage, filter to 2021/2022, and load only
  those two pinned raw source years. Read no source winner, score, duration,
  closing price, or 2023 market fields/files. Two-sided finite prices must be
  greater than 1. The known 2022 external winner error does not change prices
  or the project's original outcome labels.
- Fit the early fold on eligible earlier rows with tournament dates at least
  21 days before the earliest 2021 H2 target; validate on 2021 dates from July
  1 onward. Retain 2009–2020 rows, not just the short market history. Teacher
  match timestamps must also be no later than this fit cutoff; later teacher
  observations fall back to hard labels in both soft-target conditions.
- Independently fit the 2022 fold on prior seasons, with the same 21-day purge.
  Select one of the seven models or the R4 reference using 2021 H2 accuracy,
  provided log loss is no more than R4 +0.005; break ties by loss, then name.
  The chosen early model must improve 2022 accuracy by at least 0.005 and
  satisfy the same loss guard to upgrade the formal recommendation.
- Then select one pooled H2+2022 candidate by the same rule, refit all seven
  on 2009–2022 eligible rows, freeze the choice, and score the unchanged 2,588
  2023 targets. R4 is the reference; R5's exploratory mixture and lowest-loss
  spline are report-only controls. No mixtures or threshold tuning are added.
- Report all seven configurations, matched lambda contrasts, teacher coverage,
  full-cohort accuracy/loss/Brier, and paired tournament-date bootstrap
  intervals. These intervals are descriptive and do not account for all
  adaptive search, player dependence, or repeated reuse of the 2022 gate.

## Timing and interpretation limits

The provider describes opening quotes as prematch but supplies no quote-capture
timestamp. The recorded match UTC time is used only as a conservative upper
bound on the provider's claimed prematch information, not proof of an archived
quote publication time. Historical result dates remain tournament start weeks
under the pre-existing T−21 convention. No current-match quote is a predictor.

Falsifiers include: the teacher's useful information being chiefly late injury
news unrecoverable from the student's history; too little market-supervised
history; teacher miscalibration; and a gain reproduced by constant-label
smoothing. Beating the hard-label fit alone is insufficient evidence of
market-specific transfer. A null result does not disprove all distillation.

Reproduce: `.venv/bin/python -m scripts.train_distillation_round7`.
Frozen inference: the same module with `--predict-features CSV`; it requires
only historical features and a trusted local saved joblib, never market data.
