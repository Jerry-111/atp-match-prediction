# Round 12 proposal: jointly learned player residual states

Status: bounded design for coordinator review, **not an empirical result or a
frozen execution protocol**. Recorded 28 November 2026 after the eleven-round
results were known. No new real-data fit, validation score or final-test outcome
was inspected for this proposal. The original 2,588 targets, threshold 0.5 and
sealed 2024–2025 test remain unchanged.

## Independent generation and reopening

The research agent first generated A12, partially pooled player residuals in
the final binary likelihood, and B12, training risk robust to season/surface
environment mixtures. This happened before new literature or peer proposals.
Known anchors were the 70% goal and the eleven completed rounds. The
scientific-brainstorming workflow and its evaluation/facilitation references
were used; the participants are AI agents, not an independent human panel.
After this record was sent, the coordinator disclosed the audit agent's
similar latent-strength proposal. Agreement is not evidence of predictive
value. The audit agent then identified a specific historical residual-leakage
risk, addressed below.

**A12 hypothesis:** strong shared features still leave persistent, partially
player-specific errors. Joint shrinkage can distinguish these from a noisy run
of wins, and time coupling may improve the current residual estimate. This is
a possible improvement over fixed summaries, not a new source of information.
The strongest failure explanation is that Elo, Glicko, point strengths and
recent form already explain every stable residual that the available sample
can identify. More player parameters could then fit old noise or compete with
the shared predictors without improving future matches.

**Reopening decision:** retain A12 as three nested structures, without a new
surface-residual block, mixtures or parameter grid. Retain B12 only as a lower
priority alternative. B12's worst-group objective is not aligned directly with
the requested full-cohort average accuracy, and the checked primary evidence
includes average-accuracy losses.

## Exact novelty relative to implemented code

| Existing implementation | Difference in the proposed experiment |
| --- | --- |
| `advanced_features.py` computes Elo and smoothed `won − expected_Elo` summaries | Estimate player residuals jointly with the final feature coefficients; do not merely append another average residual feature. |
| `dynamic_features.py` implements incremental Glicko; `adjusted_point_features.py` uses online point updates | Joint conditional likelihood over all currently admissible observations and a coupled player trajectory, rather than independent incremental updates. |
| `joint_point_model.py` jointly estimates serve/return point effects | The response here is the original match-win label, with the existing supervised match-feature design in the same likelihood. No new point-count model or point-to-match approximation. |
| `matchup_features.py` adds observed-style antisymmetric interactions | Learn shrunk player-ID residual states; no learned nontransitive embeddings or new style interactions. |
| `conditional_gate.py` combines four frozen expert probabilities | Fit one conditional match model, not an expert-weight gate. |

Full latent-history modeling was discussed and deferred in Round 7. The claim
is therefore **not previously implemented here**, not a first-ever idea or a
new statistical method. A repository search plus direct inspection of the
above engines and R2/R5 fitting code found no existing joint binary
feature-plus-player-residual estimator. This is a bounded code audit, not a
claim that every theoretical variant is absent.

## Minimal matched structures and fixed priors

Load the exact `all_inputs_spline_C0.01` specification from the frozen R5
registry. It requires 199 of the original 326 input fields: 188 odd columns,
nine spline variables and 14 interactions, with C=.01, shape/interactions
scaled by .3 and a three-year sample-weight half-life. Reuse its exact
training-only transform and 256-column design, not a recreated approximation.
The three structures share the same rows, sample weights, shared penalty and
preprocessing:

1. **shared_only:** the existing R5 estimator, no player states. It must
   reproduce that exact R5 candidate's OOF and 2023 probabilities within the
   declared numerical tolerance before interpreting either extension.
2. **shared_static_residual:** jointly estimate the shared coefficients and
   one residual per observed player, with independent `u_p ~ N(0, .35²)`.
3. **shared_dynamic_residual:** use one residual per player/observed calendar
   quarter. The first state has the same `N(0, .35²)` prior. Between successive
   observed quarters, use
   `u_p,q − u_p,q_previous ~ N(0, .15² × elapsed_years)`, where elapsed years
   is the difference between quarter-start dates divided by 365.25.

There are no free states for quarters without an eligible observation. The
forecast mean/mode beyond the latest observed quarter is its last state.
Within-quarter constancy is an explicit computational approximation; it is
not an assertion that tennis skill changes only quarterly. No new surface
effect is added because the existing design already contains surface ratings
and interactions. The initial SD .35 and annual innovation SD .15 are
**pragmatic, fixed log-odds-scale priors**, not parameters recommended or
validated for tennis by the cited papers. No grid over these values is planned.

For design row `z_i` and the appropriate player states, set
`eta_i = z_i beta + u_A,i − u_B,i`. Minimize

```
sum_i w_i [log(1 + exp(eta_i)) − y_i eta_i]
  + ||beta||² / (2 × .01)
  + sum_player first_state² / (2 × .35²)
  + sum_dynamic_links (state_difference)² / (2 × .15² × elapsed_years).
```

The static structure has one first-state term per player and no dynamic
links. The three-year weights are exactly
`w_i = 2 ** (−(max(eligible_seasons) − season_i) / 3)` on the population of
the particular fit. The same state population/weights are used by static and
dynamic versions at a query. The global fit uses the same weights as R5.
This is a recency-weighted penalized likelihood (a power-likelihood MAP
adaptation), not a claim to reproduce ordinary unweighted Bayesian inference.
Dividing the entire objective and gradient by row count is allowed purely
for numerical scaling; dividing only the likelihood would change the model.

There is no intercept. Proper Gaussian priors and the positive shared ridge
penalty anchor the otherwise translation-invariant differences and give a
unique penalized solution. They do not make the decomposition causally
identifiable. Every design/state contrast reverses under A/B exchange, hence
the probability is complementary. Use the original single orientation: its
objective equals mirrored rows with half weights.

## Chronology, query updates and cold starts

For each outer prediction year 2020, 2021, 2022 and 2023:

1. Use the original pre-year training mask and 21-day purge. Fit all RMS,
   interaction centering/scales and spline transforms only on these rows.
   Jointly estimate `beta` and training player states for each extension.
   Fit the shared-only control through the unmodified R5 estimator.
2. Freeze `beta`, transforms, priors and structure for that outer year.
   A target query at nominal day T can condition player states only on
   original development matches whose nominal day is at most **T−21**,
   inclusive. Each historical row uses its unchanged precomputed features,
   which retain their own original T−21 construction.
3. Re-solve **only u**, with the shared `z_i beta` as a fixed offset, using
   all eligible observations once and the original priors. Warm starts are
   numerical starting values, not extra posterior penalties. Do not retain
   an old posterior as a new prior and also count its historical matches
   again. Query snapshots are keyed by outer year, cutoff and structure.
4. Past 2023 results may update u only after becoming T−21 eligible, as with
   the earlier rating experiments. They never update `beta`, preprocessing,
   prior values, selection, calibration or thresholds. Two targets with the
   same cutoff share the same state snapshot, regardless of query order.
5. A player absent from global pretraining may acquire a state from newly
   eligible past matches. A player with **zero eligible history** has residual
   zero; an unseen pair is still predicted from shared features. If both
   residuals are zero, prediction equals that extension's shared component.
   It need not equal the separately fitted shared-only/R4 model, because
   joint fitting can change beta. No row is dropped.

The joint optimization may revise earlier latent states using later results
**within the current admissible prefix**. This is as-of latent smoothing, not
a strict one-pass filter or a claim of no smoothing. It never conditions a
target forecast on results beyond that target's cutoff.

Historical fitted trajectories are latent parameters of the full conditional
likelihood. They must **not** be backfilled as purported historical features
for a second supervised learner, treated as honest historical forecasts or
used to train a calibration head on the same labels. An outer-year beta is
allowed to appear in the historical likelihood when conditioning states for
a later query; those in-sample historical probabilities are not OOF outputs.
Only the actual outer query forecasts are evaluated. This distinction prevents
the residual double-use/leakage failure raised by the independent reviewer.

## Smallest evaluation and falsifiers

Three structures across the three original forward years plus final 2023
give **12 global model fits**. As-of state optimizations are additional work,
not hidden inside that count. Proposed selection retains the existing
2020–2021 accuracy rule with loss at most R4+.005, then a 2022 improvement
gate of at least .005 accuracy with the same loss guard. Select among these
three structures and R4 only. R5's previous mixture is report-only; there are
no new blends. Freeze any pooled 2020–2022 exploratory choice before the one
full 2023 evaluation. The coordinator must approve the exact execution
protocol before fitting.

Report the two nested contrasts, not just the best row: static minus shared
tests useful player residuals, and dynamic minus static tests time coupling.
The mechanism predicts improvement in genuine forward full-cohort accuracy
with acceptable probability loss, rather than only smaller training loss.
Failure across the matched controls, a shrunken/negligible residual signal,
or improvements confined to fitted old trajectories counts against this
bounded approach. Worst-season/subgroup diagnostics are descriptive; they
do not change the fixed denominator, choose players, or tune priors after
looking at 2023. Earlier searches make all new 2023 claims exploratory.

## Feasibility and synthetic check

Metadata-only inspection found 19/38/36/35 query dates in 2020/2021/2022/2023:
128 possible snapshots per residual structure, or at most **256 state-only
solves**, excluding any initialization reuse. All pre-2023 development rows
contain 925 players and 9,964 observed player-quarter nodes before the exact
outer purge. This count read only season, date and player IDs, not outcomes.
The design has two nonzero state entries per match and a sparse chain prior;
dense state Hessians or full MCMC are unnecessary.

An unscored synthetic smoke used seed158, 10,000 generated rows, 256 shared
columns, 500 players, eight quarters and at most four CPU threads. L-BFGS-B
joint solves converged in 42 iterations/.029s for static and
156 iterations/.137s for quarterly states; separate fixed-offset state solves
took .002s and .028s. Five directional finite differences per structure
agreed with gradients to at most 9.05e−12; final scaled maximum gradients
were below 9.1e−8. The small scratch result is
`/tmp/atp-round12-residual-synthetic.json`, not a committed empirical artifact.
These are synthetic solver timings, **not** end-to-end real runtime estimates.
Real correlated spline inputs can converge more slowly. A reasonable initial
implementation budget is under one hour on four CPU threads, subject to a
synthetic full-size check and numerical stopping criteria before real fits.

The subsequent full-size synthetic check used 33,833 generated rows, 256
correlated columns, 925 synthetic players and 12,894 observed quarter nodes.
The first L-BFGS-B settings (`maxiter=2000`, `maxcor=20`, `ftol=1e-12`,
`gtol=1e-7`) and a stricter continuation differed by as much as .0004064 in
probability on 2,000 synthetic queries, despite zero classification changes.
Those zero changes say nothing about real-data classification stability.
Tightening to `ftol=1e-15` was still insufficient to guarantee the chosen
gradient criterion: independent zero-start full-size and a small fixture
stopped on relative function change with scaled gradients1.22e−8 and2.47e−8.
Both were correctly rejected rather than silently accepted.

Before any empirical fit, the numerical path was therefore fixed to
**L-BFGS-B, `maxiter=5000`, `maxcor=30`, `ftol=0`, `gtol=1e-10`**, requiring
optimizer success, finite parameters/objective/gradient and maximum scaled
gradient at most **1e−8**. The unchanged R5 shared control keeps its original
solver settings and additionally rejects convergence warnings/iteration-limit
termination. No prior, likelihood, sample weight or model structure changed.
No whitening or Newton-polishing branch is implemented. With these final
settings the independent full-size zero start completed in6.313s/1,993
iterations with maximum scaled gradient1.75e−9 (unscaled5.92e−5); all14
focused synthetic tests passed. The generated-only
[numerical evidence](../../results/round12/residual_synthetic_feasibility.json)
retains both the failures and successes; its generator is
`synthetic_fullsize_problem` in `tests/test_residual_state_round12.py`.
These checks establish bounded computational feasibility, not predictive gain
or a guarantee that every real-data solve will converge. Real failure must
stop the experiment under the frozen numerical criteria.

Implementation must test exact old-control reproduction, analytic gradients,
synthetic persistent and zero residual cases, static/dynamic prior limits,
target/future-label poisoning, within-date permutations, query-order invariance,
newly observed versus genuinely cold players, A/B complementarity, saved-state
reload and training-only transforms. Any numerical solver changes may improve
convergence without changing the registered objective or priors. Failed
feasibility is not permission to score a different model silently.

## Located primary evidence and limitations

- Turner and Firth (2012), [Bradley–Terry Models in R](https://www.jstatsoft.org/article/view/v048i09),
  DOI10.18637/jss.v048.i09. The [full author-hosted paper](https://wrap.warwick.ac.uk/id/eprint/47955/1/WRAP_Turner_v48i09.pdf),
  equations3–4, explicitly combines explanatory variables with normal player
  residuals. Their penalized quasi-likelihood software is not this proposed
  ridge/MAP implementation, and the example is not a tennis forecasting
  benchmark. This supports the model form, not an accuracy promise.
- Coulom (2008), [Whole-History Rating](https://www.remi-coulom.fr/WHR/WHR.pdf),
  sections2.2–2.4 and3, supplies the paired-comparison likelihood, Gaussian
  increment variance proportional to elapsed time and joint MAP principle.
  Its predictive evidence concerns Go. Shared features, quarterly nodes,
  tennis priors and recency weights here are explicit adaptations, not an
  exact WHR reproduction or transfer of its reported performance.
- Krese and Štrumbelj (2021), [A Bayesian approach to time-varying latent
  strengths](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0251945),
  DOI10.1371/journal.pone.0251945, gives direct negative tennis evidence:
  its ATP example finds no advantage over constant strengths (Table3).
  The paper explicitly addresses hindcasting and warns about forecasting
  applicability. Its restricted sample and different covariates neither
  validate nor conclusively rule out this proposal.
- B12's [group-DRO primary paper](https://arxiv.org/pdf/1911.08731),
  Sagawa et al., ICLR2020, equations2–4, supports a different training objective.
  But section3.2 reports 1–3 percentage-point average-accuracy costs in its
  early-stopped examples, while worst-group accuracy improves. Different
  environment difficulty can make this tradeoff harmful here.
  [Gulrajani and Lopez-Paz](https://arxiv.org/abs/2007.01434) additionally find
  carefully implemented ERM competitive across their domain-generalization
  benchmarks. Neither paper evaluates this tennis setting. Consequently B12
  is deferred rather than presented as a likely path to70%.

This was a bounded primary-source search on 28 November2026, after independent
ideas: queries covered Bradley–Terry covariates/random effects, tennis latent
strengths, group DRO and negative domain-generalization evidence. No academic
MCP was available, so publisher, author/institutional and arXiv texts were
checked with web search. Publisher PDF decoding and one repository timeout
were handled with accessible primary copies; no unavailable text is treated
as read. The search establishes neither systematic coverage nor global
novelty. Prior eleven-round negative results remain relevant counterevidence.
