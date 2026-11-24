# Can match history improve on ATP rankings?

How much do recent overall form, results on the current surface and prior meetings between the two players add to ATP rankings when predicting a completed men's singles match? We will compare a rank-only logistic regression with one that uses all four feature groups, then examine which combination predicts best on the 2023 validation season.

The first published modeling table provides rank difference and same-surface win-rate difference for 18,774 training and validation matches. The next teammate can use the published match inventory to derive two additional pre-match features, recent overall form and head-to-head record, and join them to these same rows. [Training and validation CSV](data/processed/model_input.csv) · [Data processing guide (中文)](docs/data-processing.md)

## Get the 20-season match inventory

**[Download the match CSV (22.7 MB)](https://raw.githubusercontent.com/Jerry-111/atp-match-prediction/refs/heads/data/shared-foundation/data/processed/mens_singles_2006_2025.csv)**. This branch also includes the file at `data/processed/mens_singles_2006_2025.csv`. Open it directly in your notebook or spreadsheet.

To rebuild the same table from the pinned sources, run the following command from the repository root with Python 3.9 or newer.

```bash
python3 -m scripts.audit_draw_coverage --first-season 2006 --export-inventory
```

The output is `data/processed/mens_singles_2006_2025.csv` with **161,989 distinct matches**. It covers main draws at the four majors, ATP Tour and Challenger events, plus available Tour and major qualifying. In 2006, qualifying covers the four majors. The 2019 and 2024 Challenger files have known gaps. The 2024–2025 outcome column is blank for final evaluation. [Annual counts, file checksum and coverage notes](DATA.md#combined-mens-singles-inventory)

Start with the [team handoff](HANDOFF.md) for the current task, verified results and ideas under discussion. Coding agents should also read [AGENTS.md](AGENTS.md). To pick up a teammate's branch or carry a new research decision into the repo, follow the [collaboration workflow](WORKFLOW.md).

## Reproduce the starter benchmark

The starter notebook uses 2015–2025 Tour and major main draws. Its ranking rule picked the winner in **1,640 of 2,588 matches (63.37%)** in 2023. Run `python3 -m scripts.prepare_data` for its training and validation feature table. [Feature and metric definitions](DATA.md#read-one-row) · [Retirement and walkover audit](OUTCOMES.md)

From the repository root, run the following command with Python 3.9 or newer. The first run downloads the annual files from TennisMyLife and checks their SHA-256 hashes. Subsequent runs reuse the local cache.

```bash
python3 scripts/baseline.py
```

The output includes the season split counts and the 2023 validation accuracy. If a source file changes, the script reports the year and stops. Review the revised data and its effect on the benchmark before updating the manifest.

The [analysis notebook](analysis.ipynb) includes the training-set figures and the validation benchmark. GitHub can display its saved outputs. To run it locally with JupyterLab, use a Python environment with `jupyterlab` and `matplotlib` installed, then run `jupyter lab analysis.ipynb` from the repository root.

## Starter notebook data and evaluation

We use the 2015–2025 ATP annual files, restricted to ATP 250, ATP 500, Masters and Grand Slam main draws on hard, clay or grass. Eligible matches have two identified players with positive rankings and a completed score. The source groups records by season and dates matches by tournament start week. A smaller ranking number indicates a better rank. Raw CSVs are downloaded directly from the provider under its educational-use terms. [Data and field definitions](https://stats.tennismylife.org/tennis-match-database) · [Source files and checksums](data-manifest.json)

| Seasons | Use | Eligible matches |
| --- | --- | ---: |
| 2015 | Initialize player history | 2,507 |
| 2016–2022 | Train and explore | 16,186 |
| 2023 | Validate | 2,588 |
| 2024–2025 | Final evaluation after model selection | 5,088 |

## Research plan and presentation

All eight planned logistic regressions include ranking and use the same matches. They cover every subset of three history features, including new matchups with no prior recorded meeting. Select by lowest 2023 mean log loss, then evaluate the frozen choices on 2024–2025. The final course submission pairs `workbook.html` with a roughly 20-minute video linked in `video_url.txt`. [Full project plan (中文)](PROJECT-PLAN.zh.md) · [Course assignment](https://docs.google.com/document/d/1w_g-yEtn9lvmXIXQWjQoS51FugM_Mx-ACh0-EZcHWR8/edit)
