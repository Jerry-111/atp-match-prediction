# Does surface history improve on ATP rankings?

We are testing whether a player's past results on the current surface improve pre-match predictions of ATP men's singles winners. The initial benchmark always picks the higher-ranked player. It correctly predicts **1,640 of 2,588 matches (63.37%)** in the 2023 validation season.

## Joining the project

Start with the [team handoff](HANDOFF.md) for the current task, verified results and ideas under discussion. Coding agents should also read [AGENTS.md](AGENTS.md). To pick up a teammate's branch or carry a new research decision into the repo, follow the [collaboration workflow](WORKFLOW.md).

## Reproduce the benchmark

From the repository root, run the following command with Python 3.9 or newer. The first run downloads the annual files from TennisMyLife and checks their SHA-256 hashes. Subsequent runs reuse the local cache.

```bash
python3 scripts/baseline.py
```

The output includes the season split counts and the 2023 validation accuracy. If a source file changes, the script reports the year and stops. Review the revised data and its effect on the benchmark before updating the manifest.

The [analysis notebook](analysis.ipynb) includes the training-set figures and the validation benchmark. GitHub can display its saved outputs. To run it locally with JupyterLab, use a Python environment with `jupyterlab` and `matplotlib` installed, then run `jupyter lab analysis.ipynb` from the repository root.

## Data and evaluation

We use the 2015–2025 ATP annual files, restricted to ATP 250, ATP 500, Masters and Grand Slam main draws on hard, clay or grass. Eligible matches have two identified players with positive rankings and a completed score. The source groups records by season and dates matches by tournament start week. A smaller ranking number indicates a better rank. Raw CSVs are downloaded directly from the provider under its educational-use terms. [Data and field definitions](https://stats.tennismylife.org/tennis-match-database) · [Source files and checksums](data-manifest.json)

| Seasons | Use | Eligible matches |
| --- | --- | ---: |
| 2015 | Initialize player history | 2,508 |
| 2016–2022 | Train and explore | 16,188 |
| 2023 | Validate | 2,588 |
| 2024–2025 | Final evaluation after model selection | 5,088 |

## Research plan and presentation

The plan defines the two logistic regressions, the historical window and the evaluation protocol. The final course submission pairs `workbook.html` with a roughly 20-minute video linked in `video_url.txt`. [Full project plan (中文)](PROJECT-PLAN.zh.md) · [Course assignment](https://docs.google.com/document/d/1w_g-yEtn9lvmXIXQWjQoS51FugM_Mx-ACh0-EZcHWR8/edit)
