# Round 7: different learning mechanisms and a separately timed information branch

Recorded 2026-11-27 before new fits. User request: change approach, consult more
papers, continue toward 70% on the complete 2,588-match 2023 cohort. The
2024–2025 final test stays sealed. Prior 2023 results and 2022 gates have been
repeatedly viewed: all further results are exploratory, including apparently
successful gates. Original task labels, cohort and 0.5 decision threshold are
fixed. Earlier results and the formal Round 4 recommendation are preserved.

## Idea provenance and decisions

Scientific-brainstorming was applied with the evaluation, responsible-AI and
facilitation references; attribution remains in `sota_evidence.md`. The
coordinator's initial commentary mentioned dynamics/matchups before independent
AI proposals; this is an explicit possible anchor. Three agents generated
proposals before reading each other's ideas. They are correlated AI tools,
not independent human experts, replications or consensus evidence. Human
domain expertise and player perspectives are absent.

Criteria were real change in information or learning mechanism, falsifiability
using matched controls, chronological validity, and bounded reproducible
compute. No subjective ratings were treated as data or automatically selected
a winner. After primary-source checks, the dispositions are:

| ID / originator | Idea and predicted improvement | Rival explanation / risk | Decision |
| --- | --- | --- | --- |
| R1 / research, P1 / coordinator | Joint dynamic latent skill may improve fixed-update ratings | Smoothing can leak future results; some ATP evidence is hindcasting with no dynamic advantage | Defer full GP/MCMC; do not portray hindcasting as forecasting or repeat Glicko under a new name |
| R2 / research, P2 / coordinator | Synthetic-task-pretrained tabular foundation model changes the fitted function family | Time distribution shift, limited context, pretrained-model provenance, memory and query transduction | Run bounded TabICLv2 contexts with same-row/same-column logistic and tree controls |
| F1 / features | Point/game outcomes provide richer training supervision than a binary label | A winner can lose total points/games; improvement may merely be label smoothing | Run mixed-outcome learning with matched uniform-soft-target controls and earlier-only calibration |
| F2 / features | Joint Bayesian serve/return likelihood uses opportunities and opponents more coherently | Costly old PyStan implementation; better point forecasts need not improve match accuracy | Defer full posterior reconstruction; maintain distinction from earlier online approximation |
| A1 / audit | Train-time market probabilities provide soft privileged supervision without market inference inputs | Teacher captures unrecoverable late information; generic smoothing explains gains | Distill only 2021–22 opening probabilities, matched uniform-teacher availability and mixing |
| A2 / audit | Verified earlier event completion may support fresher histories | Scheduled/UTC start dates do not prove result availability; suspended matches | Do not replace T−21 with an arbitrary smaller lag |
| P3 / coordinator, discussion-stage revision | Own completed earlier knockout rounds in the current edition establish prematch state without global timestamps | Small/survivor-selected histories, missing partial-match load; later information than T−21 | Separate actual-prematch branch, own player/earlier round only; explicit fallback and matched information controls |

The within-event revision uses only the player's own prior rounds, whose
completion precedes that player's next match. It does not assume that every
other player's earlier-labelled round has already finished. No future opponent,
target statistic or current-match winner can be a predictor.

## Paper search and evidence boundaries

The `nature-academic-search` multi-source-search workflow was used. Academic
MCP tools were unavailable. Its direct preflight reached CrossRef/PubMed/arXiv,
but OpenAlex discovery returned HTTP429; discovery continued through web
search to authors, journals, PMLR, arXiv and official implementation/model
cards. This is a bounded targeted search, not a systematic review. Bibliographic
and method claims are checked at primary sources; search snippets alone do
not establish a method's validity.

- Qu et al., TabICL, ICML2025: https://proceedings.mlr.press/v267/qu25d.html;
  TabICLv2 (2026): https://arxiv.org/abs/2602.11139 and official
  https://github.com/soda-inria/tabicl. Synthetic pretraining and MPS support
  make a bounded local experiment feasible; generic tabular benchmark gains
  do not imply a tennis gain. Temporal distribution shift remains a risk.
- Lopez-Paz et al., *Unifying distillation and privileged information*,
  ICLR2016, Eq4: https://leon.bottou.org/publications/pdf/iclr-2016.pdf.
  Teacher-only information may be used during learning. Market supervision
  is our adaptation, not a tennis result in that paper.
- Yuan et al., CVPR2020, *Revisiting Knowledge Distillation via Label Smoothing
  Regularization*: https://arxiv.org/abs/1909.11723. This motivates the uniform
  teacher control: softened labels can regularize without additional knowledge.
- Ingram, point-based Bayesian hierarchical tennis model, DOI
  10.1515/jqas-2018-0008: https://martiningram.github.io/papers/bayes_point_based.pdf.
  The paper models binomial points and player/surface/time components; its best
  point model still trails its Elo comparator on the reported 2014 match metrics.
  Our margin-target experiment is not a reproduction of this posterior model.
- Krese and Štrumbelj (2021), DOI10.1371/journal.pone.0251945:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8136743/. Small ATP hindcasting
  comparisons do not show the dynamic models beating their constant baseline;
  this is contrary evidence against assuming complexity must help.
- Goossens et al. (2015), *Winning in straight sets helps in Grand Slam tennis*,
  DOI10.1080/24748668.2015.11868847, author PDF
  https://fspieksma.win.tue.nl/papers/IJPAS2015paper.pdf. Previous-match set
  differences were associated with subsequent outcomes in its Grand Slam
  sample; these proxies are not direct fatigue measurements or a universal
  full-tour forecasting improvement.
- Brown and Minor, *Selecting the Best? Spillover and Shadows in Elimination
  Tournaments*, 2013 author working-paper version:
  https://www.law.northwestern.edu/research-faculty/clbe/events/roundtable/documents/BrownMinor_SpilloversShadows_6June2013.pdf.
  Earlier effort and prospective competition were associated with outcomes;
  markets already reflected those effects. We use no unknown future opponent.

None of these papers promises70%, and sample populations/information timing
are not interchangeable with this project's target matches. Queries included
dynamic paired comparison tennis, TabICL synthetic pretraining, generalized
distillation privileged information, margin supervision tennis, and previous
round fatigue/effort tennis.

## Selection and reporting

Each branch writes its exact fixed registry before fitting. Historical
branches retain T−21 inputs. Teacher prices are training-only and disclosed;
foundation weights are fixed, synthetic-pretrained and hashed. Main forward
years remain2020/21/22; the distillation branch uses2021H2 then2022 because
opening teacher coverage begins2021. Prior R5 mixtures and best-observed-loss
models are comparators only, not early-selection candidates.

Accuracy is primary with log-loss guard+0.005 against the fixed R4 reference.
Early selection uses the stated earlier fold(s); the2022 candidate must gain
at least0.005 accuracy and pass the loss guard. Pooled selection is frozen
before new2023 scoring. These are practical exploration rules, not independent
tests. Failed branches remain visible. An adoption in the later-information
branch cannot replace the formal original T−21 recommendation.

Report the full2588 denominator, accuracy/correct counts, log loss, Brier and
conditional paired uncertainty. Separate probability quality from accuracy;
70% requires1812 correct predictions. No filtering to high-confidence/later
rounds or Grand Slams can substitute for that full-cohort target.
