# Strong-model validation protocol

## Tenth round: honest conditional mixtures and historical RET controls

The original full 2,588 targets, threshold 0.5 and T−21 condition remain fixed.
Four frozen primitive forecast streams supply 6,130 OOF rows. Three meta
procedures compare equal, global and conditional weighting; only the latter
two fit parameters. Train on 2020 and select on 2021, evaluate that procedure
on 2022, then refit with all prior OOF. There is no fake 2020 meta test or
pooled reselection. The 2021 choice is R4. Global and conditional mixtures
both score 1,684 correct in 2023; conditional loss is slightly worse.

The separate I/R/RC branch shares ordinary inactivity in all three structures.
R/RC add provider-recorded past RET status, using 332/336/336 inputs respectively.
Early selection chooses R, but its 2022 gain is only 5/2,528; pooled development
selection chooses I. The 2023 counts are 1,704/1,707/1,708. No retrospective
selection of RC is made. The 30-source sample has 10 official confirmations
and 20 unresolved records, with early-era bias; limited exploration was
authorized with those uncertainties retained. RET is not a medical variable.
[Comparison and limitations](../results/round10/overview.md) ·
[Exact decisions](research/round10_protocol.md).

Rebuild the earlier audited R4/R5 and R9 primitive OOF caches first. The current
meta stage needs no new Torch fit. Use only trusted locally rebuilt bundles.

```sh
.venv/bin/python -m scripts.train_gating_round10
.venv/bin/python -m scripts.audit_gating_round10
.venv/bin/python -m scripts.train_retirement_round10
.venv/bin/python -m scripts.audit_retirement_round10
.venv/bin/python -m scripts.audit_completion_round10
.venv/bin/python -m scripts.report_round10
```

Reload the gate with `--predict-inputs results/round10/gating/inference_inputs.csv`.
Its four expert probabilities and five context columns must have the exact
frozen schema and aligned IDs. RET uses `--predict-features` and its saved
338-column union matrix, internally selecting the declared candidate columns.
The source audit is separately reproducible with `audit_retirement_source_round10`;
official PDF caches and hashes are required to recheck manual adjudications.
It does not automatically replace the human interpretation of a draw.

Six meta optimization calls plus twelve spline fits are **18 optimization
stages**, not eighteen independently selected configurations. The base suite
passes 329 tests and 97 subtests, with eight optional-Torch skips; unchanged
neural training and its isolated tests were not repeated. The completion pilot
is source/metadata-only: four of six transitions pass conservative date bounds,
three add source editions and directly cover 204/544 pilot targets. No global
lag reduction or freshness model is implemented; historical publication
evidence remains qualified. Final years stay sealed and the goal stays active.

## Ninth round: full-data neural learning and matched history controls

The next completed round tests eight structures/recipes plus15 fixed mixtures,
with all2,588 original2023 targets, threshold0.5 and T−21 history unchanged.
No new market input or supervision is added. The frozen choices score:
RealMLP/R4 mixture **1,697 /65.57% /loss0.607936**, history GRU
**1,703 /65.80% /0.608496**, and shared style matchup
**1,709 /66.04% /0.606459**. Only RealMLP passes its reused2022 branch gate,
but that gain does not persist in2023. Raw TD-S fails to beat the matched-width
plain MLP; GRU fails to beat same-input mean pooling. Preserve those results
and prior R4/R5/R7 comparator roles; no global superiority or70% is established.
[Full table and interpretation](../results/round9/overview.md) ·
[Frozen protocols](research/round9_protocol.md).

Rebuild earlier audited inputs first. Use the existing isolated
`requirements-foundation.txt` environment for Torch; keep base dependencies
unchanged. Both neural routes use CPU with four threads. All neural seeds and
fold states are retained locally; epochs are chosen by an earlier purged inner
tail and then refitted on the full outer training population.

```sh
.venv/bin/python -m scripts.train_matchup_round9
.venv-foundation/bin/python -m scripts.train_sequence_round9 --device cpu
.venv-foundation/bin/python -m scripts.train_neural_round9
.venv/bin/python -m scripts.audit_matchup_round9
.venv-foundation/bin/python -m scripts.audit_sequence_source_round9
.venv-foundation/bin/python -m scripts.audit_sequence_round9
.venv-foundation/bin/python -m scripts.audit_neural_round9
.venv/bin/python -m scripts.report_round9
```

The neural runner supports `--predict-features` and optional
`--reference-predictions`, requiring IDs aligned to the fixed R4/R5 references.
The sequence runner additionally requires `--predict-sequences`; its complete
development archive is aligned to R5 features, but inference on old training
rows is not OOF evaluation. Matchup uses its saved feature schema. Use only
trusted locally rebuilt model bundles. See branch reports for exact commands.

Base verification passes298 tests with8 optional-Torch skips and97 subtests.
The isolated Torch suite passes23 focused tests (7 overlap the base suite).
Independent audits verify36 probability columns, all48 saved neural seed/fold
states and the chronological selections. Reload errors are zero; independent
NumPy forward errors are below1.26e−7 and batch changes alter no classes.
Full gradient trajectories are not replayed. TD-S inner probe scalers were
discarded; only their required deterministic reconstruction and source path
can be checked, not comparison to an unsaved object.

The later timing audit also corrects R8's source wording: provider UTC is not
verified actual start, completion or quote publication. Every used edge still
meets the corrected nominal tournament-date T−21 guard, so no numerical
prediction changes. Do not shorten the lag using those timestamps alone.
[Evidence and completed-event feasibility](research/round9_freshness_feasibility.md).

## Eighth round: structural alternatives and active 70% goal

The user requested continued optimization and a persistent goal on 2026-11-27.
Success requires at least **1,812 correct on all 2,588 original 2023 targets**,
the unchanged 0.5 threshold, legitimate prematch inputs, audited artifacts and
reproduction. The goal remains active. Repeatedly inspected 2023 results are
exploratory even if they later cross 70%; 2024–2025 remain sealed. No
task-completion tracking automation was created.

Round 8 compares five joint point structures, ten historical event-environment
learners, six historical-market graph settings, and eighteen current-price
residual learners. Fixed blends and repeated as-of/fold fits are counted
separately. All five branch/quote-condition upgrade gates fail. The frozen
historical graph scores **66.27% / 0.607183**; the opening and latest-price
pooled residual choices score **66.96% / 0.591239** and **67.00% / 0.590046**.
The unselected opening-history setting scores **68.01% / 0.596723**, still
52 correct short and worse in loss than raw opening. It is not an adopted
upgrade. [Full comparison](../results/round8/overview.md) ·
[Protocol and evidence](research/round8_protocol.md).

After the earlier rounds, especially the audited Round 5 market files and
Round 7 event/distillation/foundation reference outputs, have been rebuilt:

```sh
.venv/bin/python -m scripts.train_joint_points_round8
.venv/bin/python -m scripts.train_venue_round8
.venv/bin/python -m scripts.train_market_graph_round8
.venv/bin/python -m scripts.train_market_residual_round8
.venv/bin/python -m scripts.report_market_residual_round8
.venv/bin/python -m scripts.audit_joint_points_round8
.venv/bin/python -m scripts.audit_market_graph_round8
.venv/bin/python -m scripts.audit_market_residual_round8
.venv/bin/python -m scripts.report_round8
.venv/bin/python -m pytest -q
```

The joint runner requires its committed synthetic timing/gradient smoke record
before any fit. It saves all 175 validation latent-state snapshots and a
frozen calibration map; reload with `--predict-targets results/round8/joint_points/inference_targets.csv`.
Unseen cutoffs require rebuilding an as-of state under the fixed protocol,
not changing calibration. The venue and graph runners support `--predict-features`
on their saved local inference matrices. The residual runner also requires
`--condition opening` or `--condition latest`. Joblib bundles and large feature/OOF
files stay local; regenerate them using these commands. Use only trusted locally
rebuilt bundles.

Historical points and market graphs may update from earlier 2023 observations
once T−21 eligible; calibration and model choices remain fixed before 2023.
The graph reads opening identity/price fields only and uses exact R4 fallback
for 176 disconnected/unknown pairs. The venue builder reads raw seasons only
through 2022 and keeps all 256 targets without venue history. Current-price
residuals have 101 exact R4 fallback rows; latest means closing if available,
otherwise opening. Opening names a quote field, not a verified decision time:
earlier-round features may postdate its capture, so that combination is a
later-prematch opening-price condition. [Timing details](research/round8_market_residual.md).

Verification: **282 tests, 97 subtests**; five independent branch audits cover
**116 probability columns × 2,588 targets**, with zero serialization error
and maximum A/B complement error 7.78e−15. The joint audit independently
reconstructs all validation raw forecasts, checks 15/175 gradients and all
20 calibration fits; it does not claim a complete independent OOF refit.
The venue and graph audits rebuild every new feature. The report-only
modules render the executed notebook/HTML without changing selections.

## Seventh round: alternative learning mechanisms

The 2026-11-27 continuation compares four explicitly separate branches:
synthetic-pretrained TabICLv2, point/game-supervised targets, training-only
market distillation, and own earlier knockout-round state. The latter uses
later prematch information than the original T−21 task; the distillation
student has historical inference inputs but additional market supervision.
Read the [protocol](research/round7_protocol.md) and
[full comparison](../results/round7/overview.md) before comparing their scores.
All retain the same 2,588 validation targets, fixed labels and threshold 0.5.
2023 and the practical 2022 gates have been reused; these are exploratory
results, and 2024–2025 remain unopened.

After rebuilding the earlier rounds and their audited local artifacts:

```sh
.venv/bin/python -m scripts.train_margin_round7
.venv/bin/python -m scripts.train_distillation_round7
.venv/bin/python -m scripts.train_event_round7
python3.12 -m venv .venv-foundation
.venv-foundation/bin/python -m pip install -r requirements-foundation.txt
.venv-foundation/bin/python -m scripts.setup_foundation_round7
.venv-foundation/bin/python -m scripts.train_foundation_round7
.venv/bin/python -m scripts.report_round7
```

Foundation inference was measured on an Apple M4 Pro with 24 GiB unified
memory. It uses MPS float32, four members, and a SHA-pinned official checkpoint;
its isolated environment does not change the base modeling dependencies.
The local fitted bundle is approximately 2.5 GB and is not committed. A
memory-scheduling repair after three CV settings, but before new 2023 scores,
is disclosed in its protocol and computational-deviation record. Rebuilding
on another device may produce small numerical differences and must be audited.

Every runner supports `--predict-features`. Margin and distillation require
the existing 326-column R5 matrix; the foundation bundle expects the 337-column
R6 transfer inference schema and internally selects its frozen 64 columns;
the event bundle expects 353 columns. Use trusted locally rebuilt joblib
artifacts. The event runner has an exact R4 fallback when neither player has
an earlier round; the feature builder validates each own knockout path and
reconciles raw duplicates before aggregating, without changing old targets.
[Source audit](research/round7_event_source_audit.md).

The selected market-supervised student scores **1,712/2,588 = 66.15%** with
**0.604599** log loss, compared with the same hard-label learner's 0.606453.
Its conditional paired loss-improvement interval spans zero
[-0.000299, 0.003986]; this is a promising probability comparison, not a
confirmed gain or a 70% result. Heavy uniform smoothing reaches 1,716 correct
as an unselected observation but worsens loss to 0.622271. Preserve this
negative probability result rather than selecting it after viewing 2023.

The foundation branch passed its predeclared 2022 gate but its frozen choice
scores only 1,686/2,588 (65.15%, loss 0.614779) on 2023. Keep the frozen choice
and negative result as recorded; do not retrospectively change its blend.
The original pure-history probability recommendation remains a separate R4
reference. Branch-local adoption rules do not establish global superiority.

Adopted 2026-11-24 under the user's request to optimize beyond the initial
course baseline. The claim to test is improved probability estimation on the
same completed ATP Tour/major matches. This is a strong-model search, not an
established SOTA result. The [evidence register](research/sota_evidence.md)
records independent ideas, primary literature, contrary evidence and deferred
directions. The original eight logistic subsets remain the explanatory core.

## Reproduce

From the repository root, use Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-modeling.txt
.venv/bin/python -m scripts.train_models --with-serve
.venv/bin/python -m scripts.model_report
.venv/bin/python -m pytest -q
```

The first modeling run downloads only pinned 2006–2023 annual sources for
historical point statistics. Raw sources stay in the ignored local cache.
Subsequent runs reuse checksum-verified files. The published inventory and
four-feature table supply all other inputs. The model runner does not fetch
2024–2025 raw files or score their outcomes.

## Decisions frozen before the new validation scores

- Targets: all 16,186 training rows (2016–2022) and 2,588 validation rows
  (2023), identical to the original task. Outcomes describe normally completed
  matches with eligible ranks, not all scheduled matches including retirements.
- Histories: eligible Tour/major main draws from 2006; each query includes
  dates at most T−21. Tournament start dates are proxies for actual match
  dates, so the lag is deliberately conservative. Past validation-season
  results may enter later historical features, without refitting the model.
- Forward folds: train on earlier seasons, predict 2020, 2021 and 2022 in
  turn. Fit rows must also precede each fold's earliest tournament date by
  at least 21 days. Scalers, coefficients and trees are fitted on that fold's
  training rows only. No random train/validation split or random early-stopping
  split is used.
- Original comparison: eight subsets always include log rank difference;
  each chooses C from 0.01, 0.1, 1, 10 using pooled forward-fold log loss.
  Select the original subset by 2023 log loss, breaking ties by fewer inputs.
- Additional inputs: overall/surface Elo at K=16,32,64; surface shrinkage;
  90/364/728-day form and Elo residuals; history counts; inactivity; H2H;
  rank transforms; surface context; shrunk historical serve/return rates.
  Elo begins at 1500, base-10 scale 400. All updates sharing a date use
  pre-date ratings and are added together, avoiding an invented match order.
- Models: regularized symmetric logistic models, CatBoost, histogram gradient
  boosting and uncalibrated overall Elo. The full machine-readable grid is
  written to `results/modeling/search_registry.json` before model fitting.
- Exchangeability: every training observation is paired with swapped players
  and a complemented outcome, each at half weight. Prediction is averaged
  with one minus its swapped prediction. This guarantees P(A beats B) equals
  1−P(B beats A), including nonlinear models with symmetric context.
- Ensemble: select each family's settings and convex weights using 2020–2021
  OOF predictions; compare the blend procedure with the earlier-selected
  best single model on 2022. If that gate passes, refit family selection and
  blend weights using all three folds. Otherwise retain the best pooled-CV
  single model. The final ensemble OOF score is a meta-training score and
  must not be described as independent validation.
- The advanced recommendation is persisted before 2023 scoring. Every
  selected family and original subset is reported, even if the frozen
  recommendation is not the best 2023 row. There is no post-validation tuning.
- Report log loss, accuracy and Brier score; paired date-block bootstrap
  intervals compare log loss against ranking logistic regression. They are
  exploratory, conditional on fitted models, and do not account for all
  selection uncertainty or dependence from players across dates.

## Artifacts and handoff

`results/modeling/validation_comparison.csv` is the common-denominator result
table. `validation_predictions.csv` contains match IDs and each reported
probability. `temporal_cv.csv` retains every search result. The frozen JSON
records exact family choices, ensemble weights, lag, seed and source hashes;
the search registry records feature names and hyperparameters.

The locally generated `frozen_models.joblib` contains all fitted reported
models. Load only this trusted locally generated artifact; its classes have
importable module paths. It is reproducible from the command above and is
not tracked in Git. The two large intermediate feature/OOF tables are also
generated locally and ignored. Pinned Python packages are recorded in
`requirements-modeling.txt`.

To predict from a prepared matrix with the frozen feature definitions, without
refitting or reading its labels:

```sh
.venv/bin/python -m scripts.predict_frozen --features path/to/features.csv --output path/to/predictions.csv
```

The predictor checks the required numeric feature schema and preserves an
optional `match_id` column. Independent artifact reload reproduced all
17 × 2,588 validation probabilities exactly; the shared scorer reproduced all
saved metrics within 1e−12. The complete suite passes 45 tests.

The next evaluator should preserve these model parameters, rebuild the same
feature schema for the final test rows under the same date cutoff, and score
2024 and 2025 once. The development feature builders intentionally reject
final-test targets; extending that gate and feeding eligible past test results
is an explicit evaluator step, not a new model search. Do not retrain on 2023
without declaring a different protocol. Historical updates and parameter
refitting are different operations.

The report and notebook distinguish measured validation findings from ideas.
The follow-ups below evaluate skill uncertainty, graph/centrality ratings,
score-margin Elo and mechanistic point models. Historical odds or injury/news
features still require verified availability timestamps and a separate
information regime.

## Follow-up paper experiments

The original results above remain archived. Reproduce the subsequent rounds
in order, because each preserves and compares the previous frozen artifacts:

```sh
.venv/bin/python -m scripts.train_round2
.venv/bin/python -m scripts.report_round2
.venv/bin/python -m scripts.train_round3
.venv/bin/python -m scripts.report_round3
.venv/bin/python -m scripts.train_round4
.venv/bin/python -m scripts.report_round4
```

Round four adds paper-derived margin ratings, opponent-network centrality,
standardized point-to-match probabilities, and age/ranking-point features to
the existing 202-column matrix. It retains the 33,833 expanded training rows
and 2,588 validation rows. Separate age and points controls help identify the
source of profile improvements. The [protocol](research/round4_protocol.md)
records paper adaptations, snapshot corrections, interrupted runs and the
51-candidate search. The [report](../results/round4/report_summary.md) records
the outcome; [the executed notebook](../round4.ipynb) and
[standalone HTML](../round4.html) contain the same saved evidence.

All follow-up 2023 scores are exploratory: the season has already informed
earlier work. Round four must improve the current round-three early-selected
2022 reference by at least 0.001 before replacing its procedure. The gate is
a practical selection rule, not independent confirmation or a significance
test. The final 2024–2025 test remains unopened.

Reload the locally generated fourth-round bundle without fitting:

```sh
.venv/bin/python -m scripts.train_round4 --predict-features path/to/prepared_features.csv
```

This writes `results/round4/inference_predictions.csv`. Inputs must follow the
frozen feature definitions and include all 259 numeric features; labels and
extra columns are ignored. Raw model probabilities, prior recommendations and
the new family finalists are retained. Model files and large intermediate
matrices are generated locally rather than committed. The final evaluator
must deliberately extend the development-only target guards, preserve all
coefficients and cutoff rules, and compare frozen procedures on identical
test matches.

The completed round-four audit reloaded all 21 reported probability columns
on all 2,588 matches exactly, reproduced shared log loss/accuracy/Brier exactly,
and verified A/B complementarity to 2.22e−16. A separate reviewer reproduced
all 204 CV measurements and all selections. The round-four suite passed 113 tests
and 31 subtests; [machine-readable audit](../results/round4/artifact_audit.json).

## Accuracy-focused fifth round

The user next requested progress toward 70% accuracy. The [combined result
table](../results/round5/overview.md) records that this target was not reached.
The same 2,588 completed 2023 matches are scored throughout, with a fixed 0.5
threshold. No candidate is credited with a result on a filtered easier subset.

After reproducing rounds one through four, run:

```sh
.venv/bin/python -m scripts.diagnose_round5
.venv/bin/python -m scripts.train_round5
.venv/bin/python -m scripts.report_round5
.venv/bin/python -m scripts.train_market_round5
.venv/bin/python -m scripts.report_market_round5
.venv/bin/python -m scripts.closing_blend_round5
.venv/bin/python -m scripts.round5_overview
.venv/bin/python -m pytest -q
```

The historical branch adds 67 numeric inputs, for 326 total. They summarize
97,764 eligible Challenger/qualifying matches and past serve/return style,
height and handedness. Fitting still uses 33,833 target matches; historical
matches supply features, not additional target labels. Feature queries retain
the T−21 cutoff. Seventy-five new configurations compare feature subsets,
recency, spline and random Fourier interactions, logistic and squared-hinge
losses, and boosted trees. Three earlier configurations supply cached OOF
controls. The [protocol](research/round5_protocol.md) describes fixed grids,
paper adaptations, missingness and calibration assumptions.

Early 2020–2021 accuracy selection must improve 2022 accuracy by at least
0.5 percentage points, while increasing log loss by at most 0.005. The gain
was only 5 / 2,528 = 0.198 percentage points, so the formal historical model
remains the fourth-round age spline. The pooled-CV exploratory mixture reaches
1,715 / 2,588 = 66.27%, with log loss 0.60987. The lowest observed new log loss
is 0.60645 from the all-input spline at 65.92% accuracy; it is a separate
probability comparator, not the accuracy-selected recommendation.

Reload the historical bundle with all 326 required numeric features:

```sh
.venv/bin/python -m scripts.train_round5 --predict-features path/to/prepared_features.csv
```

This writes `results/round5/inference_predictions.csv`. The bundle includes
the earlier frozen recommendation and the new comparisons; it does not refit
on prediction input. Large feature, OOF and joblib files remain local and
ignored. Committed registries, selections, validation probabilities, reports
and audits make the experiment reviewable.

### Separate market-information experiments

The opening-price branch uses 20 fixed candidates and pinned public
[Valuebetennis data](https://www.valuebetennis.com/en/donnees.htm), attributed
under CC BY 4.0 in the [source audit](research/round5_market.md).
`data/market-manifest.json` records URLs and checksums for 2021–2023 only.
Keep the verified raw cache: source URLs are mutable and a checksum mismatch
must stop reproduction rather than silently use different data. Matching reads
no winner or score fields and preserves every target row. Opening prices are
available for 2,487 validation rows; the other 101 predictions fall back
exactly to the historical model.

This branch uses actual prematch prices as described by the supplier, a later
information condition than the original T−21 task. Quote capture timestamps
are unavailable, so their historical availability cannot be independently
verified. Do not present market gains as improvements from historical features.
The [market protocol](research/round5_market_protocol.md) records 2021-H2
selection, the 2022 gate, and calibration using only earlier observations and
true historical-model OOF probabilities. Its selected opening fusion reaches
66.73%, compared with 66.77% for the raw opening-price baseline.

The additional 16 fixed closing-price controls were proposed after seeing the
main and opening-price 2023 scores. Their weights use 2022 only, but this is
explicitly adaptive exploration and upgrades no recommendation. The selected
blend reaches 67.27%, below the raw closing-price baseline's 67.70%; 111 missing
closing quotes fall back to the specified historical base. The latter is a
market benchmark, not a trained model achievement. The opening experiment
excludes closing prices from training and selection.

For inference, `scripts.train_market_round5.predict_market_bundle(bundle,
history, features)` accepts frozen historical probabilities and the aligned
eight-column market matrix. `scripts.closing_blend_round5.predict_closing_controls`
applies the recorded fixed blends without fitting. The evaluator must preserve
source matching, probability orientation, missing-price fallback and each
information regime, alongside the historical feature schema.

Independent audits reproduced all 24 historical, 21 opening/market and 16
closing-control probability columns on all 2,588 matches, with maximum
round-trip error 2.22e−16. They also checked temporal selection, every OOF
candidate and missing-price fallback; the market audit replayed calibration
fits. See the [historical audit](../results/round5/artifact_audit.json),
[market audit](../results/round5/market/artifact_audit.json) and
[closing-control audit](../results/round5/closing_blend/artifact_audit.json).
The round-five suite passed **167 tests and 58 subtests**. All 2023 scores
remain exploratory after repeated inspection. The 2024–2025
test seasons remain unopened; extending development-only guards belongs to
the final evaluator, with frozen model choices and no new search on that test.

## Sixth round: new information and supervised transfer

The [sixth-round overview](../results/round6/overview.md) preserves a negative
result: neither new historical information nor the tested auxiliary
supervision improves the previous best 2023 accuracy. All comparisons retain
2,588 targets, T−21 and a fixed 0.5 threshold. The original final test is sealed.

After rounds one through five, run:

```sh
.venv/bin/python -m scripts.audit_round6_labels
.venv/bin/python -m scripts.train_round6
.venv/bin/python -m scripts.report_round6
.venv/bin/python -m scripts.train_transfer_round6
.venv/bin/python -m scripts.probability_control_round6
.venv/bin/python -m scripts.round6_overview
.venv/bin/python -m pytest -q
```

The main run adds 57 columns to the untouched 326-column matrix: 20 official
event-history ratings, 17 rank-trajectory descriptors and 20 breakpoint/hold
summaries. The new rating banks include 4,291 additional conventional completed
official matches and matched Tour-only banks; exhibition/alternate-format
events are excluded by explicit rules. Thus an entire-block effect cannot
be attributed solely to extra matches. Historical ranking peaks are only
observed since 2006; conflicting player/date snapshots are masked. Breakpoint
residuals are not identified psychological ability. The
[protocol](research/round6_protocol.md) and individual feature notes record
all priors, eligibility rules and limitations.

The 34 fixed configurations use the prior 2020/2021/2022 forward folds. Only
the current Round 4 reference is eligible as a previous selection candidate;
Round 5's pooled mixture and lowest observed loss model are reported solely
as frozen comparators. The 2022 gain of 0.396 percentage points falls short
of 0.5. The resulting exploratory mixture scores 65.42% / 0.60774 on 2023,
so the formal recommendation stays unchanged.

The separate [transfer protocol](research/round6_transfer.md) uses up to
75,261 extra Challenger/qualifying supervised rows and a compact 48-column
shared representation. Twelve settings compare auxiliary weights 0/.1/.3/1,
with 36 fixed mixtures with Round 4. Scalers/bases see only positive-weight
training rows, weights normalize to the Tour training count, and every fold
retains the 21-day purge. The full positive-weight final fits use 109,094 rows;
the selected zero-auxiliary model fits only 33,833 Tour rows. Its selected
mixture scores 65.77% / 0.60928 and does not pass the upgrade gate.

After the main accuracy results were seen, a single additional comparator
selects the minimum pooled-CV log loss among the same 34 main configurations.
That explicitly adaptive comparison scores 66.07% / 0.60755 and upgrades no
recommendation. It is not a new hyperparameter grid or independent validation.

Reload without fitting:

```sh
.venv/bin/python -m scripts.train_round6 --predict-features path/to/383-column-features.csv
.venv/bin/python -m scripts.train_transfer_round6 --predict-features results/round6/transfer/inference_features.csv
```

Transfer inference expects 337 columns: the previous 326 plus 11 additional
compact descriptors, with shared multilevel fields checked for equality.
The probability-control bundle stores the selected model and its feature
names for direct `model.predict_proba(prepared_features)` inference. All
joblib files and large feature/target/OOF matrices remain local and ignored.

The independent [source audit](../results/round6/source_audit.md) deliberately
reads external winners **after** R5's outcome-blind links are frozen. It
never changes matches, features or labels. One 2022 external winner error is
resolved using an official ATP draw in favor of the original project label;
all 2,502 matched 2023 winners agree, and the 86 unmatched targets remain
unchecked by that source. This diagnostic cannot explain or erase model errors.

Independent artifact audits cover the [main experiment](../results/round6/artifact_audit.json),
[transfer experiment](../results/round6/transfer/artifact_audit.json) and
[adaptive probability control](../results/round6/probability_control/artifact_audit.json).
They check saved probabilities, full-cohort metrics, chronological selections
and feature/inference contracts. Prior artifacts and predictions remain intact.
The completed six-round suite passes **207 tests and 91 subtests**.

## Eleventh round: bounded completed-event freshness

The [fixed protocol](research/round11_freshness_protocol.md) tests whether a
small, evidenced addition of more recent completed-event results helps two
unchanged algorithms. All 2,388 / 2,528 / 2,588 targets in 2021 / 2022 / 2023
remain, with threshold 0.5. The original T−21 history is the stale control;
fresh queries add only disjoint eligible match IDs from individually qualified
completed source editions. This is a separate, later-information experiment,
not an improvement under the original T−21 information set. No current target
edition, new ranking, technical statistic, auxiliary match or extra target is
introduced. The 2024–2025 final test remains sealed.

The source sample is the first ten editions by nominal date and lexical ID in
each of 2021, 2022 and 2023, fixed without consulting individual prediction
errors. The [qualification ledger](../results/round11/completion/qualified_pairs.json)
admits 15 source–target pairs using nine distinct source editions. Their target
editions contain 168 / 183 / 234 rows; direct source-player overlap covers
42 / 90 / 118 rows. These are coverage counts, not reduced accuracy denominators
or claims about the number of changed predictions. Unqualified, overlapping and
unreviewed pairs remain recorded; targets without added history return their
own stale input and probability exactly.

Qualification is **Tier B under explicit public-result availability, stability
and schedule-consistency assumptions**. Let D be the later of the supported
completed-by day and displayed publication day. Its conservative upper bound,
D at 00:00 UTC plus 36 hours, must be strictly before the independently
supported target lower day at 00:00 UTC minus 14 hours. Unknown publication
timezones remain unknown. A readable completed-final report supports the
edition under the stated assumptions; it neither certifies every historical
row nor makes today's page an immutable as-of snapshot. Official event-period
starts are lower bounds under schedule consistency, not observed match clocks.
See the [independent evidence review](research/round11_availability_review.md).

The two paired algorithms are K=32 overall Elo and a shared rank/rating
logistic model. The latter uses only `log_rank_diff`, `elo_k32_diff` and
`shrunk_surface_elo_k32_diff`, C=.01, no intercept, training-only scaling and
mirrored half-weight rows. There is one stale-only fit for each prediction
year, with training from 2009 and the original 21-day purge: **three fits in
total, four new probability columns**. Each fitted scaler and coefficient
vector is applied unchanged to both stale and fresh inputs. This tests
transport to fresher inputs, not a pipeline retrained on fresh covariates.
Elo updates retain nominal-date batches and the original 20-match surface
shrinkage; there is no invented within-event ordering or tuning sweep.

Only 2021 selects the algorithm: fresh accuracy must increase and log loss
may worsen by at most .005 against its own stale control. The single selected
algorithm must then gain at least .005 accuracy on all 2022 rows under the same
loss guard. Choice and gate are saved before the 2023 fit and evaluation; there
is no pooled reselection. R4 and R5 are contextual references excluded from
this mechanism selector. All 2023 results remain repeatedly inspected
exploration, and the reused 2022 gate is development evidence, not an
independent confirmation.

The completed run selects **no freshness algorithm**: on 2021, fresh Elo loses
three correct predictions and the shared logistic has no accuracy change, so
neither satisfies the first-stage rule. The gate therefore fails and the R4
historical recommendation remains. The 2022 descriptive paired changes are
+4 correct for Elo and +1 for logistic; they do not trigger a new selection.
On the full 2023 cohort:

| Algorithm | Stale correct / 2,588 | Fresh correct / 2,588 | Stale log loss | Fresh log loss |
| --- | ---: | ---: | ---: | ---: |
| Overall Elo | 1,640 (63.37%) | 1,642 (63.45%) | 0.639146 | 0.638342 |
| Shared rank/rating logistic | 1,678 (64.84%) | 1,678 (64.84%) | 0.622797 | 0.622267 |

Both methods change probabilities on 118 rows. Elo flips two classifications,
both becoming correct; logistic flips none. The 2,354 rows without extra
history reproduce their own stale predictions exactly. The date-block paired
intervals include zero; they are conditional exploratory intervals without
search correction. The modest loss decreases do not establish a broad
freshness benefit, and neither new method exceeds R4 (1,696 correct) or the
frozen R5 exploration (1,715). Reaching 70% still requires at least 1,812 correct.
The bounded source coverage cannot answer whether a complete season of
independently supported fresh history would help. Numerical details are saved
in [matched controls](../results/round11/freshness/matched_controls.csv),
[the frozen choice](../results/round11/freshness/frozen_selection.json) and
[the summary](../results/round11/freshness/summary.json).

The actual run started before an independent pre-score request to pause for
an integer net-correct tie fix reached the runner. It used the original
floating-mean selector, whose exact source snapshot is retained. Only
`choose_procedure` subsequently changed; both versions select no method on
this run's −3 / 0 gains. All 17 original artifact hashes remain unchanged,
and no model or probability was refitted or modified. The
[selection correction record](../results/round11/freshness/selection_correction.json)
documents this execution-order deviation. The completed base suite passes
343 tests and 109 subtests, with eight optional-dependency skips; see
[verification](../results/round11/verification.json). Independent artifact
audit completion is recorded separately rather than inferred from those tests.

After rebuilding the existing environment and prerequisite artifacts through
Round 5, reproduce into a new directory rather than overwrite the frozen run:

```sh
.venv/bin/python -m scripts.train_freshness_round11 \
  --output /tmp/atp-round11-reproduction \
  --approved-protocol-sha256 690c60196e3b8700157036c5632e5b17190c86c102843530d7f48c972eed1ce7 \
  --approved-manifest-sha256 54281191164bdcfae2cc3cc71cc53db080653d4b6137686f960581517b10dc9d
.venv/bin/python -m scripts.audit_freshness_round11
.venv/bin/python -m scripts.report_round11
```

The first command checks both approved hashes and all ledger-bound inputs
before fitting. No fresh web lookup is required. The audit and report commands
default to the original `results/round11/freshness` artifacts; they do not
automatically switch to the reproduction directory. Keep the existing frozen
outputs intact. Current source uses the corrected integer selector; the
original run's source is preserved in its correction record. Saved models are
`model_2021.joblib`, `model_2022.joblib` and
`frozen_models.joblib`; the matrices and probabilities retain every target row.

## Twelfth round: conditional binary player residuals

This pure-history experiment keeps the full2,588 targets, threshold.5 and
T−21 availability. It reuses the exact R5 `all_inputs_spline_C0.01` transform
and compares shared-only, static player residuals, and quarterly dynamic
residuals. Shared coefficients and states are jointly fitted on each outer
training population; later within-year queries freeze coefficients and
preprocessing and condition only states on the eligible prefix. The initial
SD.35 and annual innovation SD.15 are fixed pragmatic log-odds priors, not
validated ATP hyperparameters from the papers. No market or charting data enter.

The shared-only control reproduces all6,130 OOF and2,588 final development
probabilities exactly. There are12 global optimization stages and256 separate
state solves. The early selector chooses static residuals, but its2022 gain is
only4 correct over R4, below the fixed13-match gate. The pooled exploratory
choice is dynamic; gate failure preserves R4. On the full2023 cohort:

| Model | Correct /2,588 | Accuracy | Mean log loss |
| --- | ---: | ---: | ---: |
| Original shared control |1,706|65.9196%|.606453|
| Static player residuals |1,711|66.1128%|.608872|
| Dynamic player residuals |1,692|65.3787%|.611061|

Both extensions worsen matched log loss in every forward year. Static's
five additional correct predictions have a date-block interval containing
zero; dynamic loses14 against the same control. No candidate reaches1,812.
All repeated2023 comparisons remain exploratory;2024–2025 stay sealed.
[Full report](../results/round12/overview.md) ·
[Protocol](research/round12_residual_state_protocol.md) ·
[Primary literature and limits](research/round12_residual_state_proposal.md).

Rebuild the existing environment and R5 prerequisite artifacts, then use a
new directory so the original outputs remain intact:

```sh
.venv/bin/python -m scripts.train_residual_state_round12 \
  --output /tmp/atp-round12-reproduction \
  --approved-protocol-sha256 54fbc7f95d8419cde22d0d26a1f14d5f6e01acd748a72f73959f365dacc45e34
.venv/bin/python -m scripts.audit_residual_state_round12
.venv/bin/python -m scripts.report_round12
```

The latter commands default to the original frozen artifacts, not the temporary
reproduction. Pre-fit readiness binds the reviewed code/protocol; the run
registry binds inputs, and selection is written and hashed before2023 scoring.
There is one fixed numerical solver, with a separate strict gradient check;
failed synthetic precision trials remain documented. No failed real fit was
silently replaced. Saved `model_2020.joblib` through `model_2023.joblib` and
`frozen_models.joblib` include complete states and eligible identities, and
are local reproducible artifacts rather than committed binary weights.

For an already saved cutoff, inference requires the original finite feature
columns plus a separate label-free CSV with `match_id`, `season`,
`tourney_date`, `player_a_id`, `player_b_id`:

```sh
.venv/bin/python -m scripts.train_residual_state_round12 \
  --predict-features /absolute/path/to/features.csv \
  --query-metadata /absolute/path/to/query_metadata.csv
```

This writes `inference_predictions.csv` and performs no optimization. The
`predict_asof` engine API can build a new permitted cutoff from explicitly
provided, unchanged2009+ historical features and labels; it filters by cutoff
before reading history outcomes. Existing snapshots reject changed eligible
history. Query years are bounded to the bundle's frozen development year;
this command does not authorize or run the sealed final evaluation.
