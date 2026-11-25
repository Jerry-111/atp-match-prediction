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
Deferred candidates include graph/centrality ratings, explicitly modeled
skill uncertainty, score-margin Elo and mechanistic point models. Historical
odds or injury/news features require verified availability timestamps and a
separate information regime. None of these is asserted to improve results
without a matched experiment.
