# Round 12 research decision record

Recorded before Round 12 empirical fitting, 28 November 2026. The user owns
the objective: improve the unchanged full 2,588-match 2023 forecast toward
1,812 correct at threshold .5, with defensible information timing and a
reproducible saved model. 2024–2025 remain sealed. Earlier eleven-round
results and the target accuracy are known anchors. This is exploratory
development, not a new untouched validation experiment or a SOTA claim.

The coordinator applied the scientific-brainstorming skill. Research,
features and audit agents generated proposals before receiving peers' new
ideas or doing a new literature search. They are AI collaborators sharing
prior context, not an independent human expert panel. Following generation,
we checked existing implementations, primary sources, rival explanations and
feasibility. We used qualitative criteria: full-cohort reach, distinctness
from completed experiments, temporal validity, an interpretable matched
control, bounded computation, and what a negative result would teach. No
invented vote or weighted numerical preference is reported.

| Initial idea / origin | Evidence or objection after generation | Decision |
| --- | --- | --- |
| Joint final-match feature coefficients and partially pooled player residuals / research A12, independently audit A12-1 | Existing Glicko is a sequential rating feature; R8 jointly models point outcomes, not this conditional binary likelihood. There is a plausible persistent residual signal, but strong old features may already explain it. | Implement three nested structures: shared-only, static residual, quarterly dynamic residual. |
| Group distributionally robust training / research B12 | Worse-group risk differs from the user's full-cohort accuracy target. Checked primary examples include an average accuracy cost; small tennis environments could be noisy. | Defer; do not add an objective grid to the residual test. |
| More lower-level market-supervised examples from the cached source / features F12-A | Strict source checks find incomplete/truncated scores and selection bias. Simple monotone opening-price calibration cannot change the predicted winner. Extra rows alone do not establish a mechanism for correcting the remaining errors. | Complete a source feasibility report; do not fit this proposal. A future proposal needs reliable labels and a distinct legal ranking signal. |
| Charting tactical features / features F12-B | Potentially distinct shot-level information, but historic publication availability and sparse player coverage need evidence. | Defer acquisition and fitting; a known match date is insufficient proof of prior publication. |
| Handedness and height matchups / audit A12-2 | Direct code review finds these in R5 and further R9 style interactions. | Retracted as a new direction. |
| Kernel / hinge classification / coordinator R12-1 | R5 already tests odd random Fourier features and squared hinge heads. | Drop as a duplicate. |
| Ranking trajectories / coordinator R12-2 | R6 already has 90/365-day trends, peaks, volatility, missingness and gaps. | Drop as a duplicate. |
| Multiple-bookmaker consensus / coordinator R12-3 | Distinct from the single cached bookmaker, but the inspected external provider limits automated/AI data use and its data pages returned 403. No data were downloaded or denial bypassed. | Defer; source access is not established. |
| Replaying already-eligible knockout history by round / audit follow-up | Existing rating engines batch whole nominal tournament dates. Round metadata may provide partial order without inventing exact timestamps. This is a model assumption, not a proven bug. RR requires its original batch treatment. | Retain as a later, separate paired K32 experiment; no fitting in the residual branch. |

Read-only audit found no new common target alignment or scoring fault:
36,421 unique development targets, 33,833 pre-2023 fitting rows and 2,588
2023 targets; no canonical A/B violations. Late-December next-season dates
are legitimate rather than automatic leaks. A small number of historically
unranked players does not support a large missing-population explanation.
These checks do not establish that every old feature or source label is correct.

The retained hypothesis and strongest rival are registered in the
[residual-state proposal](round12_residual_state_proposal.md). Its primary
references support covariates plus player effects and Gaussian temporal
coupling; they do not predict 70% here. A separate
[ATP dynamic-model working paper](https://papers.tinbergen.nl/18009.pdf)
supports final-probability residual updates, but its restricted player
population and different years prevent direct accuracy transfer. Ordinary
one-state score-driven BT can be an Elo reparameterization; naming it
"dynamic" alone is not novelty. The proposal also retains negative tennis
evidence on latent trajectories.

Implementation authority is the user's existing optimization request. The
coordinator freezes the exact protocol and source hashes only after synthetic
tests and independent agent review. This internal readiness check does not
require a new user approval. Real scores must not be computed before readiness;
if a numerical or source issue requires a change, record it before resuming.
No change in cohort, threshold, information regime or final-test embargo is
authorized by a disappointing result.
