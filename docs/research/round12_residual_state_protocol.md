# Round 12: conditional binary player residuals

Status: fixed execution protocol, recorded before empirical results. Bind this
file's SHA-256 in the readiness record before execution. The runner requires the approved hash and a fresh
output directory. Its registry also binds the engine, runner, input artifacts
and selection rule before any real-data fit.

## Population and information

Use only the original R3 development target table (2009–2023, 36,421 rows)
and R5 historical feature matrix (326 unchanged columns). Verify the pinned
artifacts and row identities. No raw 2024–2025 data are opened, and no market,
new completion-date or charting information enters this branch. The residual
history starts in 2009 because earlier rows lack this exact shared design;
the old features retain their original earlier warm-up history.

Outer prediction years are 2020, 2021, 2022 and 2023. `temporal_masks` retains
only earlier seasons dated at or before the first target date minus 21 days
for global fitting. Train-only transforms and shared coefficients then freeze.
For a target at T, state-only conditioning may use development observations
dated at or before T−21, including eligible earlier matches in that prediction
year. Query metadata contain no target outcome. Filter history before reading
its outcomes or features. No history outside 2009–2023 is accepted.

Every target remains, including unknown players. An individual with no
eligible observations receives residual zero; one absent from the global
training vocabulary may acquire a state from subsequently eligible history.
Observed-prefix latent smoothing is allowed; it is not strict filtering.
States never become backfilled training features for another learner.

## Fixed models and optimization

Exactly three structures share the frozen R5 `all_inputs_spline_C0.01`
specification and transform: 199 required inputs, 188 odd main columns,
nine odd quadratic spline inputs, 14 context interactions, design width256,
C=.01, shape/interaction scales .3 and three-year season weighting.

1. `shared_only`: fit the unmodified R5 spline estimator.
2. `shared_static_residual`: jointly fit beta and one Gaussian residual per
   player with initial SD .35 log odds.
3. `shared_dynamic_residual`: jointly fit beta and player/calendar-quarter
   states. Initial SD .35; innovations have variance .15² times the elapsed
   years between observed quarter starts (365.25 days/year). Missing quarters
   have no free nodes. The latest eligible state is the forecast residual.

Use the exact weighted sum binary log likelihood plus beta ridge
`||beta||²/(2*.01)`, initial-state penalties and the stated increment penalties.
The [proposal](round12_residual_state_proposal.md) gives the full formula.
Both state extensions reuse the already fitted control's train-only transform;
its coefficients are only an optimization start, not a fixed final beta.
There is no extra intercept, surface state, prior grid or calibration.
The fixed prior scales are pragmatic adaptations, not paper-derived ATP values.

At each distinct target cutoff, hold beta fixed and solve only states with
the same original priors and all eligible observations counted once. Query
weights use the same three-year formula with the maximum eligible season.
Warm starts, if used, change starting points only. Save node identities,
states, eligible-ID hashes, cutoff, beta, transforms and solver diagnostics.
No Bayesian posterior probability integration is claimed: forecasts use
penalized maximum-a-posteriori log odds.

There are 12 global optimization stages (three structures × four years),
plus at most 256 state-only solves across 128 distinct outer-year dates.
As-of solves are disclosed separately, not called three total fits. Solver
failure stops the run before selection/scoring. Numerical fixes may preserve
the same objective, but must be documented; changing priors, inputs or model
structures requires a new exploration record. Do not silently discard failures.

The residual optimizer is L-BFGS-B with maxiter=5000, maxcor=30, ftol=0
and gtol=1e−10, applied to the entire objective and gradient divided by row
count. Require optimizer success and a finite scaled maximum absolute
gradient no greater than1e−8; preserve the unscaled value too. The original
shared-only sklearn solver remains unchanged for exact control reproduction.
Before any empirical fit, a full-size synthetic sensitivity check showed
that the initial looser solver could differ by about4.1e−4 in probability
from a stricter solve. This justified numerical tightening without changing
the objective, data, priors or candidates. No real tennis labels informed it.
An intermediate ftol=1e−15 setting still stopped just above the gradient
limit on both a small fixture and an independent full-size zero start; those
failed feasibility checks are retained. Setting ftol=0 passed both, including
full-size scaled maximum gradient1.75e−9. There is no Newton-CG, whitening or
adaptive alternative solver path. Success still requires the separate
gradient acceptance check, because floating-point function equality can
otherwise stop L-BFGS-B before its requested gradient tolerance.

## Selection before 2023 scoring

The candidate set is the three structures and the frozen R4 reference.
For 2020–2021 pooled forward predictions, exclude candidates whose mean log
loss exceeds R4 by .005. Among eligible candidates choose the greatest
integer number correct at threshold .5, then lower mean log loss, then
lexicographic name. No probability blend or threshold search occurs.

The early choice passes the 2022 gate only with at least 13 additional correct
out of 2,528 (the ceiling of .005×2,528), and loss at most R4+.005. Independently
record the same single-model selection on pooled 2020–2022. If the early gate
passes, recommend the pooled choice; otherwise retain R4. Write and hash this
decision before producing/scoring 2023 probabilities. The pooled choice is
still reported when the gate fails. Reuse of this gate across rounds is
exploratory and is not independent statistical confirmation.

The old R5 frozen accuracy exploration, rank baseline and raw Elo may appear
as contextual 2023 comparators. They are not extra selection candidates.
No 2023 result overrides the frozen choice. Success requires at least
1,812/2,588 correct and the complete audit/reproduction handoff, rather than
an attractive partial cohort or an unsupported SOTA label.

## Tests, audit and interpretation

Before interpreting the extensions, reproduce all old `all_inputs_spline_C0.01`
OOF and 2023 probabilities to maximum absolute tolerance 1e−10, with identical
threshold decisions. Also require synthetic analytic-gradient checks, state
prior behavior, cold/new player handling, A/B complementarity, query-order
invariance, cutoff exclusion and saved-state reload. The independent review
must explicitly acknowledge readiness before empirical fitting.

Keep every candidate's full-cohort predictions, model states, coefficients,
fit populations, source hashes and convergence records. Independently verify
the saved probability calculation, the original control, row alignment,
selection, matched counts/losses, temporal prefix identities and future-label
poisoning. Audit limitations must distinguish reloading fitted states from
replaying optimizers or rebuilding the complete historical feature matrix.

Primary contrasts are static minus shared, and dynamic minus static; dynamic
minus shared is a useful additional descriptive contrast. Report paired
date-block accuracy/loss intervals using the existing bootstrap procedure.
These condition on fitted predictions and do not account fully for shared
players or repeated model search. Record the magnitude of residual log-odds
contributions without claiming causal interpretation. A lower training loss
alone does not support useful future player residuals.

The sources, negative evidence, rejected duplicates and AI collaboration
limits are in the [proposal](round12_residual_state_proposal.md) and
[decision record](round12_decisions.md). Preserve original results even when
this experiment is negative. 2024–2025 stay sealed for the next evaluator.
