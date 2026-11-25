# Strong-model validation protocol

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
The completed suite passes **167 tests and 58 subtests**. All 2023 scores
remain exploratory after repeated inspection. The 2024–2025
test seasons remain unopened; extending development-only guards belongs to
the final evaluator, with frozen model choices and no new search on that test.
