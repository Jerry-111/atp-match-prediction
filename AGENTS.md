# Working in this repository

This is a four-person course project about predicting ATP singles winners. The same notebook will supply the figures, code walkthrough and results for peer grading.

## Start a session

1. Read [HANDOFF.md](HANDOFF.md) for the latest verified result, the next task and any ideas still under consideration. Follow its issue or PR link to see who is working on it.
2. Read the relevant part of [the research plan](PROJECT-PLAN.zh.md) for the adopted design. [WORKFLOW.md](WORKFLOW.md) explains how to carry a decision from a chat into the shared repo.
3. Run `git status -sb` and `git fetch origin`. On a clean local `main`, use `git pull --ff-only` before branching. If resuming a teammate's work, use the branch or PR in their handoff and preserve local edits.

## Work with the evidence

- Reuse `load_matches`, `split_name` and `ranking_result` from [scripts/baseline.py](scripts/baseline.py). The source manifest pins the annual files. Each collaborator downloads the provider's raw CSVs locally through the loader.
- Use `python3 -m scripts.prepare_data` or `build_model_rows` for the shared training/validation table. [DATA.md](DATA.md) defines its columns and metric denominators. The [outcome audit](OUTCOMES.md) documents what changes when a mid-match retirement counts as a loss.
- Use the season splits defined by the loader. Treat the tournament start week as the prediction cutoff. The plan defines the lag for past results. Assign player A/B independently of the winner/loser columns before fitting a model.
- Keep model selection on 2023. The evaluation teammate scores 2024–2025 after the validation settings are frozen. Label dataset counts, validation measurements and final test measurements explicitly.
- When a change affects results, rerun the affected computation and refresh saved notebook outputs before quoting numbers. For a documentation change, verify the links and claims it touches.

## End a session

Put work someone else can pick up on a shared branch or in a PR. Give the next teammate the link, the result or check they can reproduce, and the next action. Update [HANDOFF.md](HANDOFF.md) when a milestone, decision or next task changes. A routine code edit can use the PR description alone.
