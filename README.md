# Can match history improve on ATP rankings?

**Eleventh round: bounded fresher history gives a small probability gain, no model upgrade.** Two paired algorithms and three shared logistic fits keep every original target. Under explicit retrospective publication and schedule assumptions, 15 qualified source–target pairs affect 118 player-history rows in 2023. Overall Elo improves from **1,640 to 1,642/2,588** (loss **0.639146 to 0.638342**); rank + overall/surface Elo stays at **1,678** (loss **0.622797 to 0.622267**). Neither algorithm gained accuracy on the 2021 selection cohort, so no freshness procedure is adopted. Missing evidence preserves exact stale fallback. The previous highest unselected market observation remains **1,760/2,588**, still 52 short of 70%; final years remain sealed and the goal remains active. [Round 11 comparison](results/round11/overview.md) · [Executed report](round11.ipynb) · [Frozen protocol](docs/research/round11_freshness_protocol.md)

**Tenth round: conditional routing and historical RET do not close the accuracy gap.** The four-expert gate scores **1,684/2,588**, identical in correct count to its global-weight control; 2021 selection retains R4. Three nested inactivity/RET models score **1,704 / 1,707 / 1,708**, with losses **0.606294 / 0.605856 / 0.605707**. Earlier pooled selection chooses ordinary inactivity, and the RET upgrade gate fails. The RET source sample has **10 officially verified and 20 unresolved** records; this is a limited analysis of recorded status, not injuries. There are six meta optimizer runs and twelve spline fits, with no new blend grid. A separate six-transition completion-source pilot adds no model score. The 70% goal remains active; 2024–2025 remain sealed. [Round 10 comparison](results/round10/overview.md) · [Executed report](round10.ipynb) · [Decision record](docs/research/round10_protocol.md)

**Ninth round: no accuracy breakthrough.** Full-data RealMLP-TD-S, ordered player histories and antisymmetric style interactions test **8 fitted structures/recipes plus 15 fixed mixtures**. The development-frozen choices score **1,697 / 65.57%**, **1,703 / 65.80%** and **1,709 / 66.04%**, respectively, on all 2,588 original matches. RealMLP passes its reused 2022 gate but does not carry that gain into 2023; GRU does not beat the same-input mean-pooling control. Preserve these negative results. Independent audits verify all **36 probability columns**, saved neural folds and preprocessing. The base suite passes **298 tests and 97 subtests**; the isolated Torch suite passes 23 focused tests. The goal remains active and 2024–2025 sealed. [Round 9 comparison and reproduction](results/round9/overview.md) · [Executed report](round9.ipynb) · [Protocol](docs/research/round9_protocol.md)

**Eighth round (2026-11-27): the 70% goal remains active.** Joint serve/return skills, historical event environments, historical bookmaker graphs and current-price residuals add **39 structural/parameter settings** under separate information conditions. None passes its development-period upgrade gate. The frozen historical-market graph scores **1,715/2,588 (66.27%), loss 0.60718**. The highest current-price observation is **1,760/2,588 (68.01%)**, but it was **not selected** and has worse log loss than raw opening prices; it is still 52 correct predictions short of 70%. All targets and the 0.5 threshold remain unchanged. **282 tests and 97 subtests pass**, and independent audits verify all 116 new-branch probability columns. 2024–2025 remain sealed. [Round 8 comparison and reproduction](results/round8/overview.md) · [Executed report](round8.ipynb) · [Research protocol](docs/research/round8_protocol.md)

**Seventh round (2026-11-27): a probability improvement to verify; 70% remains unreached.** Four paper-informed branches test **35 fitted settings**: TabICLv2, point/game supervision, training-only market distillation and own earlier knockout-round state. The frozen market-supervised student uses only historical inputs at inference and scores **66.15% / 0.60460 log loss**, versus its identical hard-label control's **65.92% / 0.60645**. The conditional improvement interval includes zero. TabICLv2 passed its historical gate but fell to **65.15%** on 2023; its frozen negative result is retained. The later-information event branch fails its gate, scoring **65.42%** for its pooled exploration. All comparisons retain 2,588 matches; 2024–2025 remain sealed. [Round 7 table and reproduction](results/round7/overview.md) · [Executed report](round7.ipynb) · [Paper and decision record](docs/research/round7_protocol.md)

**Sixth round (2026-11-27): no accuracy upgrade.** Round 6 added 57 historical inputs and ran **34 main configurations**, **12 supervised-transfer configurations**, and **36 fixed transfer blends**. The development-selected new history mixture scores **65.42%**; the transfer procedure scores **65.77%** and selects zero auxiliary weight. An explicitly adaptive probability comparator reaches **66.07% / 0.60755**. None beats the previous historical exploration's **66.27%** or lowest observed loss **0.60645**. The formal Round 4 model remains; all 2,588 targets, T−21 history and the sealed final test are preserved. [Round 6 comparison and reproduction](results/round6/overview.md) · [Executed report](round6.ipynb) · [Independent source audit](results/round6/source_audit.md)

**Accuracy-focused follow-up: 70% not reached.** Round 5 added 67 inputs using 97,764 eligible Challenger/qualifying history matches and historical technical composition, then compared **75 new historical-model configurations**, **20 opening-market configurations**, and **16 explicitly adaptive closing-price controls**. On all 2,588 matches, the best historical exploration reaches **66.27%**; the added-market exploratory fusion reaches **67.27%**. The separate raw closing-market benchmark with historical fallback scores **67.70%** (still 60 correct matches short of 70%). The historical upgrade gate failed, so the formal Round 4 recommendation remains. Market inputs are a later information condition, and no model is shown to beat the market. [One comparison table and reproduction](results/round5/overview.md) · [Historical report](round5.ipynb) · [Separate market report](round5_market.ipynb)

**Paper-informed follow-up (2026-11-25).** Round 4 tested **51 configurations** adapted from primary papers on weighted Elo, opponent networks, point-to-match probabilities and age/ranking-point features. The new training-period-selected model adds age representations to the previous spline model: **0.60941** log loss / **65.53%** accuracy on the same 2,588 exploratory 2023 matches. Ranking loss is **0.62980**, a **3.24%** relative reduction. The best observed new comparison scores **0.60868**, but is not substituted for the selected model. The incremental gain over Round 3 remains uncertain; 2024–2025 stays unopened and no SOTA claim is made. [Round 4 results](results/round4/report_summary.md) · [Paper protocol](docs/research/round4_protocol.md) · [Executed report](round4.ipynb) · [Current handoff](docs/agents/handoff.md)

**Earlier follow-ups.** Rounds 2 and 3 tested **69 new configurations** and expanded fitting to **33,833** matches. Round 3 selected an expanded-history spline at **0.61082 / 65.53%**; expanded linear remains an accuracy comparator at **65.88%**. All earlier results are preserved. [Round 2](results/round2/report_summary.md) · [Round 3](results/round3/report_summary.md)

**Modeling results (2026-11-24).** The training-period-selected ensemble reduces 2023 validation log loss from **0.62980** (ranking logistic) to **0.61153**, with accuracy increasing from **63.37%** to **65.07%** on the same 2,588 matches. The richer linear comparator has the lowest observed validation loss, **0.61056**. Final 2024–2025 testing remains unopened; these results do not establish SOTA. [Results and reproduction](docs/modeling.md) · [Complete comparison](results/modeling/validation_comparison.csv) · [Executed modeling notebook](modeling.ipynb)

How much do recent overall form, results on the current surface and prior meetings between the two players add to ATP rankings when predicting a completed men's singles match? We compare a rank-only logistic regression with all four feature groups, then extend the experiment with strength ratings, historical point statistics and nonlinear models.

Start with the [modeling protocol](docs/modeling.md) and [team handoff](docs/agents/handoff.md) to reproduce [modeling task #1](https://github.com/Jerry-111/atp-match-prediction/issues/1). The [four-feature modeling CSV](data/processed/model_features.csv) contains all 16,186 training and 2,588 validation matches, the outcome, four pre-match predictors and the historical counts behind them. Rebuild it offline from the published tables with `python3 -m scripts.prepare_features`. [Data processing guide (中文)](docs/data-processing.md)

## Get the 20-season match inventory

**[Download the match CSV (22.7 MB)](https://raw.githubusercontent.com/Jerry-111/atp-match-prediction/refs/heads/main/data/processed/mens_singles_2006_2025.csv)**. The file is also available locally at `data/processed/mens_singles_2006_2025.csv`. Open it directly in your notebook or spreadsheet.

To rebuild the same table from the pinned sources, run the following command from the repository root with Python 3.9 or newer.

```bash
python3 -m scripts.audit_draw_coverage --first-season 2006 --export-inventory
```

The output is `data/processed/mens_singles_2006_2025.csv` with **161,989 distinct matches**. It covers main draws at the four majors, ATP Tour and Challenger events, plus available Tour and major qualifying. In 2006, qualifying covers the four majors. The 2019 and 2024 Challenger files have known gaps. The 2024–2025 outcome column is blank for final evaluation. [Annual counts, file checksum and coverage notes](docs/data.md#combined-mens-singles-inventory)

Project documentation lives in [docs/](docs/). Start with the [team handoff](docs/agents/handoff.md) for the current task, verified results and ideas under discussion. Coding agents should also read [AGENTS.md](AGENTS.md). To pick up a teammate's branch or carry a new research decision into the repo, follow the [collaboration workflow](docs/agents/workflow.md).

## Reproduce the starter benchmark

The starter notebook uses 2015–2025 Tour and major main draws. Its ranking rule picked the winner in **1,640 of 2,588 matches (63.37%)** in 2023. Run `python3 -m scripts.prepare_data` for its training and validation feature table. [Feature and metric definitions](docs/data.md#read-one-row) · [Retirement and walkover audit](docs/outcomes.md)

From the repository root, run the following command with Python 3.9 or newer. The first run downloads the annual files from TennisMyLife and checks their SHA-256 hashes. Subsequent runs reuse the local cache.

```bash
python3 scripts/baseline.py
```

The output includes the season split counts and the 2023 validation accuracy. If a source file changes, the script reports the year and stops. Review the revised data and its effect on the benchmark before updating the manifest.

The [analysis notebook](analysis.ipynb) includes the training-set figures, validation benchmark and executed model comparisons. GitHub can display its saved outputs. Install `requirements-modeling.txt` plus `jupyterlab` to run interactively, then run `jupyter lab analysis.ipynb` from the repository root. The standalone [workbook.html](workbook.html) displays the saved results without a Python environment; final-test results and the video remain pending.

## Starter notebook data and evaluation

We use the 2015–2025 ATP annual files, restricted to ATP 250, ATP 500, Masters and Grand Slam main draws on hard, clay or grass. Eligible matches have two identified players with positive rankings and a completed score. The source groups records by season and dates matches by tournament start week. A smaller ranking number indicates a better rank. Raw CSVs are downloaded directly from the provider under its educational-use terms. [Data and field definitions](https://stats.tennismylife.org/tennis-match-database) · [Source files and checksums](data-manifest.json)

| Seasons | Use | Eligible matches |
| --- | --- | ---: |
| 2015 | Initialize player history | 2,507 |
| 2016–2022 | Train and explore | 16,186 |
| 2023 | Validate | 2,588 |
| 2024–2025 | Final evaluation after model selection | 5,088 |

## Research plan and presentation

All eight fitted logistic subsets include ranking and use the same matches, including new matchups with no prior recorded meeting. A standalone Elo benchmark and richer models are reported on the same validation set. The original logistic subset is selected by 2023 mean log loss; the advanced recommendation is selected using training-period folds before 2023 scoring. Next, evaluate the frozen choices on 2024–2025. The final course submission pairs `workbook.html` with a roughly 20-minute video linked in `video_url.txt`. [Full project plan (中文)](docs/project-plan.zh.md) · [Course assignment](https://docs.google.com/document/d/1w_g-yEtn9lvmXIXQWjQoS51FugM_Mx-ACh0-EZcHWR8/edit)
