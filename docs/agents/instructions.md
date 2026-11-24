# Working in this repository

This is a four-person course project about predicting ATP singles winners. The same notebook will supply the figures, code walkthrough and results for peer grading.

## Start a session

1. Read [the team handoff](handoff.md) for the latest verified result, the next task and any ideas still under consideration. Follow its issue or PR link to see who is working on it.
2. Read the relevant part of [the research plan](../project-plan.zh.md) for the adopted design. [The team workflow](workflow.md) explains how to carry a decision from a chat into the shared repo.
3. Run `git status -sb` and `git fetch origin`. On a clean local `main`, use `git pull --ff-only` before branching. If resuming a teammate's work, use the branch or PR in their handoff and preserve local edits.

## Work with the evidence

- Start with the [four-feature CSV](../../data/processed/model_features.csv). Rebuild it offline with `python3 -m scripts.prepare_features` from the published inventory and starter feature table. The [data guide](../data-processing.md) specifies the A/B orientation, windows, smoothing and a worked match.
- The shared 2006–2025 collection is built with `python3 -m scripts.audit_draw_coverage --first-season 2006 --export-inventory`. Check the 161,989-row count and [published checksum](../data.md#combined-mens-singles-inventory) when exchanging versions. The commands below apply to the starter Tour main-draw experiment.
- Reuse `load_matches`, `split_name` and `ranking_result` from [scripts/baseline.py](../../scripts/baseline.py). The source manifest pins the annual files. Each collaborator downloads the provider's raw CSVs locally through the loader.
- Use `python3 -m scripts.prepare_data` or `build_model_rows` for the shared training/validation table. [The data reference](../data.md) defines its columns and metric denominators. The [outcome audit](../outcomes.md) documents what changes when a mid-match retirement counts as a loss.
- Use the season splits defined by the loader. Treat the tournament start week as the prediction cutoff. The plan defines the lag for past results. Assign player A/B independently of the winner/loser columns before fitting a model.
- Fit the eight prespecified rank-containing subsets on identical 2016–2022 rows. Compare rank-only and all four groups as planned. Select the subset with the lowest 2023 mean log loss and report accuracy alongside it. Score a separate overall Elo benchmark on the same matches using the same T−21 historical cutoff. The evaluation teammate scores 2024–2025 after the settings are frozen. Label dataset counts, validation measurements and final test measurements explicitly.
- When a change affects results, rerun the affected computation and refresh saved notebook outputs before quoting numbers. For a documentation change, verify the links and claims it touches.

## End a session

Put work someone else can pick up on a shared branch or in a PR. Give the next teammate the link, the result or check they can reproduce, and the next action. Update [the team handoff](handoff.md) when a milestone, decision or next task changes. A routine code edit can use the PR description alone.
