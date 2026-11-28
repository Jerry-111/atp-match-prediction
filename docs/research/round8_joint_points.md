# Round 8: jointly fitted serve/return skills

Status: design reviewed and authorized by root before empirical fitting. Owner: research agent.
Decision owner: root. Date: 2026-11-27. No Round 8 outcome scores have been
computed when this record is written. Earlier 2023 results are already known;
this is exploration, never retroactive preregistration or final-test evidence.

## Independent origin and evidence

The independent round first proposed (L1) a joint point-generating model,
(L2) a shared player-history sequence encoder, and (L3) separate completion /
retirement prediction. These ideas preceded this round's literature search and
other agents' suggestions. Existing seven-round failures were an acknowledged
anchor. After sharing, other agents independently proposed joint point skills;
agreement is not evidence of predictive benefit.

Primary source: Martin Ingram (2019), *A point-based Bayesian hierarchical model
to predict the outcome of tennis matches*, JQAS 15(4), 313–325,
[DOI 10.1515/jqas-2018-0008](https://doi.org/10.1515/jqas-2018-0008),
[author's full paper](https://martiningram.github.io/papers/bayes_point_based.pdf).
Equations 1–2 specify a binomial service-point likelihood with separate server /
returner skills, a shared player-surface preference, and a tournament intercept.
Equations 3–11 specify hierarchical Gaussian priors and random-walk evolution.
Section 2.2.3 fits NUTS and averages scoring probabilities over posterior draws.
Table 2 reports 68.8% / 0.592 on 2,208 selected 2014 matches, versus Elo's
69.5% / 0.586. Retirements and missing-statistic matches were excluded in that
paper. It is direct evidence for feasibility and a counterexample to assuming
point models beat Elo. Its 80-minute fitting cost motivates approximation here.

Our implementation is **joint penalized binomial MAP with fixed priors and
recency weights**, not a reproduction of NUTS, the random-walk posterior,
estimated hyperpriors or posterior integration. The online diagonal updates
already used in Round 2 are not this algorithm: here all eligible historical
observations jointly determine all coefficients to convergence at each cutoff.

Other located primary evidence, not selected for this experiment:

- [Dumovic and Howarth, Stanford CS230 2018 project](https://cs230.stanford.edu/projects_spring_2018/reports/8290687.pdf)
  used a shared 128-state LSTM over 50 previous matches and reported 69.6% versus
  a 65.6% rank control. It required at least 25 past matches per player and did
  not clearly establish a chronological split, so its percentage is not a
  comparable benchmark. L2 remains deferred, rather than discredited.
- [Oliver et al., 2024](https://doi.org/10.1002/ejsc.12177) found 3,539 retirements
  among 167,211 ATP matches (2.11%) in its historical cohort. This descriptive
  association study does not establish usable pre-match forecasting skill.
  Rare events and the 21-day lag constrain L3's likely standalone impact.

Search log: 2026-11-27, public primary-source web retrieval after independent
generation. Queries included `tennis serve return dynamic Bayesian hierarchical
model point probabilities prediction paper`, `tennis match prediction recurrent
neural network player history sequence forecasting paper`, and `tennis retirement
match prediction probabilistic model retirement risk paper`. Academic MCP tools
were unavailable; the skill's OpenAlex fallback returned HTTP 429. Author,
publisher, university and arXiv sources were checked directly. This bounded
search establishes neither novelty nor an exhaustive evidence review.

## Population, information and safety boundaries

- Identical 2,588 targets from 2023; canonical A/B identities, original labels,
  missing-data retention and fixed decision threshold 0.5 remain unchanged.
- Only pinned annual Tour sources for seasons 2006–2023 may be opened. The
  published eligible Tour inventory supplies history identities and dates.
  Source joins validate IDs, dates, surfaces and years before using point counts.
- A paired point observation requires both players' service counts to pass the
  existing `_counts` checks. Missing point data exclude an observation from the
  point likelihood, never a target from prediction. Full source and omission
  counts are exported. The target's own score/statistics are never a predictor.
- For target tournament-start date T, the as-of cutoff is T−21 inclusive.
  The common observation pool is [T−21−1460 days, T−21]. All models use this same
  pool. Source dates within a batch are order-independent. Queries never obtain
  a state fitted at a later cutoff.
- Latent skills may assimilate earlier 2023 results only after they become
  T−21 eligible, exactly as historical ratings do. Structural settings,
  calibration, selection and mixture weights cannot use any 2023 labels.
  No 2024/2025 annual raw file, test outcome or test metric is opened.
- Unknown players/surface combinations/tournament codes receive zero deviations
  under their priors. They still receive a finite probability, with no ranking
  or experience-based target exclusion.

## Likelihood and the five structural settings

Each past match produces two service observations with integer successes y and
opportunities n. For server i, returner j, surface s and source tournament code v,

```
eta = logit(.65) + intercept + surface_baseline[s]
      + serve[i] - return[j]
      + preference[i,s] - preference[j,s] + tournament[v]
loss = sum_m w_m * (n_m * logaddexp(0, eta_m) - y_m * eta_m)
       + 0.5 * sum_k (coefficient_k / prior_sd_k)^2
```

Missing terms for a control are omitted, not fitted and then ignored. The
fixed Gaussian prior SDs are: intercept 0.5, surface baseline 0.2, serving 0.2,
returning 0.15, shared player-surface preference 0.08, tournament 0.08.
These are fixed conservative logit-scale regularizers, not estimated posterior
SDs or a hyperparameter grid. Hard is the reference surface baseline; other
surface offsets and all player deviations are regularized. Tournament identity
is the tuple (year-stripped source code, Unicode-normalized/case-folded tournament
name, surface). This prevents known reused codes (424 and 8998) from pooling
unrelated events. Name changes may reduce pooling and then use the neutral prior;
no manual alias is inferred from future records. No future observation determines
the feature vocabulary.

| Name | Opponent return skill | Recency | Player-surface preference | Event intercept |
| --- | --- | --- | --- | --- |
| `serve_only_recent` | No | 364-day half-life | No | No |
| `joint_static` | Yes | Uniform | No | No |
| `joint_recent` | Yes | 364-day half-life | No | No |
| `joint_surface` | Yes | 364-day half-life | Yes | No |
| `joint_surface_event` | Yes | 364-day half-life | Yes | Yes |

All settings include the same global and surface baselines. Exponential weights
are `2**(-age_days/364)` without renormalization: older evidence has less weight
relative to a fixed prior. Report this intentional distinction, not a hidden
change to an arbitrary sklearn C. No point-count cap or label smoothing is used.
Sparse CSR design matrices and analytic L-BFGS gradients fit the convex joint
objective. A failed convergence check is a computational failure to resolve,
not a reason to remove difficult targets or pick a different configuration.

Predicted serve probabilities feed the existing, independently tested
`paper_point_model.match_probability`, including its standardized scoring rule:
advantage games and a seven-point win-by-two tiebreak at 6–6 in every set.
Known best-of-three/five format is retained. That standardized final-set rule
differs from some real historical and 2023 Grand Slam rules; it is a declared
approximation. No target score is used to infer format. We use MAP plug-in
probabilities, not a claimed Bayesian posterior predictive integral.

## Honest calibration, controls and selection

Generate raw as-of probabilities for every eligible Tour target in 2017–2022.
For each outer fold 2020/2021/2022, fit only a positive, no-intercept calibration
slope to earlier-season as-of log odds, with the same 21-day purge before the
earliest fold date. Final 2023 calibration uses only eligible 2017–2022 rows.
The calibration input for every row was computed without that row's observation
or any later observation. A positive slope preserves raw 0.5 classifications;
it can improve calibration and affects mixtures. Calibration is per structural
setting, not selected by 2023 performance.

Five calibrated settings plus each one's fixed R4 probability mixtures with
new-model weights .25/.5/.75 give **20 new selectable controls plus R4**.
These are five structural fitting settings, not 20 separately optimized point
models. Raw uncalibrated metrics are diagnostic only and never selectable.

Use the established accuracy-first rule: 2020–2021 chooses among 21 entries,
subject to log loss ≤ R4 + .005; ties resolve by lower loss then stable name.
The early choice must improve 2022 accuracy by ≥ .005 with loss increase ≤ .005
to pass the branch gate. Pooled 2020–2022 then fixes the comparison choice before
2023 inference/scoring. A failed gate retains R4; a passed gate is branch-local,
not proof of superiority after many rounds. R5 accuracy/probability comparators
and R2 `adjusted_linear_C0.003` are reported but excluded from selection.

The old R2 comparator includes a different learned feature system, training
history and classifier, so it is **not** a matched method-only control. The
five nested models above isolate opponent adjustment, recency, surface and event
effects more cleanly. All four paired contrasts are reported, including negative
results, in each CV year and separately in 2023. Same-data direct rate versus
joint adjustment is tested by the serve-only control; it does not recreate all
Round 2 inputs.

## Cost, adversarial checks and deliverables

Before empirical fitting, run unscored synthetic tests: finite-difference gradient
agreement; recovery of known opponent-adjusted effects; deterministic reordering;
player-swap complementarity; no change under poisoned current/future statistics;
and unknown-player priors. Time a ~10,000-service-observation sparse fit, with
at most four CPU threads, before starting the actual as-of runs. No MPS required.
Expected working memory is below 2 GB; time is uncertain until this smoke test.
Use warm starts by stable coefficient keys, retaining the same optimum. If the
full design is impractical, record a computational deviation before outcomes;
do not silently substitute the existing one-pass diagonal update.

Falsifiers: joint adjustment fails its matched control; apparent gains vanish
against an otherwise identical recency model; added surface/event terms worsen
forward-fold loss; all benefit is from calibration rather than classifications;
or a few low-history players dominate improvements. IID point errors, unmodeled
retirement, event-code changes, plug-in overconfidence and selection reuse remain
plausible limitations. None justify altering the denominator or labels.

Export source hashes, raw as-of probabilities, cutoff/state metadata, convergence
diagnostics, calibration rows/slopes, frozen selection, all 2023 comparisons,
matched controls, a reloadable state/calibration bundle and a concise report.
The primary question is whether jointly revisiting historical point observations
adds useful information beyond existing ratings—not whether a familiar method
can be renamed to look new. Threshold, complete cohort and final-test seal are
non-negotiable gates. Review owner/root must approve this concrete protocol
before empirical fitting. Root reviewed and authorized the five settings and
21-entry menu, requiring the event-identity correction above before fitting.
