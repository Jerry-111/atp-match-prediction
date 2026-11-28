# Strong-model exploration and evidence register

Date: 2026-11-24 (America/Los_Angeles). Decision owner: the project user. Research role: Codex research agent; proposals below are AI-assisted. This is a bounded search and an experiment proposal, not a claim of state-of-the-art performance. The user authorized a broader optimization experiment than the original eight logistic subsets. The original comparisons remain necessary to answer the course question.

## Question and constraints

Can additional pre-match information improve probability forecasts beyond ATP rankings, and which feasible model extracts that information most reliably?

Keep the target population, match IDs, A/B orientation, completed-match eligibility and T−21 cutoff identical across comparisons. Keep 2024–2025 outcomes sealed for the evaluation teammate. Select settings using expanding training-period folds; use 2023 for the documented validation comparison. Broad searching on 2023 would make the winner increasingly optimistic, so report the number of candidates and retain every tested result. Later within-season outcomes may enter later history only after their availability cutoff; they may not train or tune the frozen predictor.

This work uses public sports records. No clinical, animal, biosafety, or sensitive personal-data review is implicated. Exact match timestamps, trustworthy injury/news timestamps, historical odds, and detailed tracking data are not established inputs. Those are dependencies, not features that can simply be assumed available.

## Initial independent ideas

These entries were recorded before this agent's literature search. The agent saw the existing project question and its baseline design, but did not see other agents' proposed solutions. No human participant round occurred; this limits perspective diversity. All entries are **ideas**, not findings.

| ID | Candidate | Assumption and prediction | Disconfirming result or main uncertainty |
| --- | --- | --- | --- |
| I1 | Multiple-timescale overall and surface Elo; shrink sparse surface estimates toward overall ability | Opponent strength and surface preference contain complementary signal; reduced future log loss | Surface component adds no gain after overall skill; early-career initialization dominates |
| I2 | Lagged serve/return point rates, pooled by opportunities and shrunk for sparse history | Point outcomes distinguish dominant wins from narrow wins and stabilize estimates | Missingness or opponent mix explains apparent gains; point rates add no fold-consistent information |
| I3 | Opponent-adjusted recent form: observed wins minus pre-match expected wins | Surprising performance captures changes sooner than raw win percentage | Residuals merely add noise or repeat Elo; gain vanishes after Elo ablation |
| I4 | Regularized tabular boosting on rating differences, historical rates and known context | Nonlinear effects and interactions improve on additive logistic regression | Train loss improves while rolling validation worsens; feature order introduces asymmetry |
| I5 | Temporal out-of-fold probability calibration and a small blend | Components make different errors; calibration reduces overconfidence | Calibration overfits a small sample or blend gives negligible improvement |
| I6 | Dynamic hierarchical skill with uncertainty/inactivity, potentially surface-specific | Sparse players benefit from partial pooling and explicit uncertainty | Complexity adds unstable estimates, tuning burden, and no out-of-time benefit |
| I7 | A time-stamped market-probability benchmark | A market forecast contains additional contemporary information | Odds were collected after the prediction cutoff, match joins fail, or margin removal is inconsistent |
| I8 | Sequence/graph neural representations of past opponents and performance | Learned representations capture interactions omitted by hand-built features | Too little effective independent data; gains disappear under strict temporal validation |

Cluster relation: I1/I3/I6/I8 estimate evolving ability; I2 estimates performance mechanisms; I4/I5 combine information; I7 defines an external information benchmark. These ideas are not merged because their assumptions differ.

Before examining new validation outcomes, prioritize by (1) no future-information access, a mandatory gate; (2) mean out-of-time log loss, lower is better; (3) consistency across years, lower variance is preferable; (4) coverage of the same matches; (5) reproducibility and implementation cost. Accuracy and Brier score are secondary diagnostics. There is no numerical creativity score or vote that can establish scientific superiority.

## Located primary evidence

Each summary describes what its source reports, not a replication in this repository. Sources were checked on 2026-11-24.

| Source | Located evidence | Implication and comparison limit |
| --- | --- | --- |
| Kovalchik (2016), *Searching for the GOAT of tennis win prediction*, [DOI 10.1515/jqas-2015-0059](https://doi.org/10.1515/jqas-2015-0059), [primary full text](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2015-0059/pdf?licenseType=free) | Compared 11 forecasting approaches on 2,395 ATP matches in 2014; ranking regression and a dynamic Elo approach were strong candidates. Performance varied by player ranking. | Elo is a substantive benchmark. Published accuracy on another season or rank stratum is not a threshold for this project's SOTA claim. |
| Vaughan Williams, Liu and Gerrard, *How well do Elo-based ratings predict professional tennis matches?*, [2019 author working paper](https://www.ntu.ac.uk/media/documents/research-documents/how-well-do-elo-based-ratings-predict-professional-tennis-matches.pdf); [2021 journal DOI 10.1515/jqas-2019-0110](https://doi.org/10.1515/jqas-2019-0110) | On Wimbledon 2018, men's betting odds performed best on accuracy/calibration; a weighted overall/surface Elo composite performed best on discrimination. | Test an overall/surface blend and distinguish probability calibration from classification accuracy. One grass tournament does not establish general ATP gains. |
| Kovalchik (2020), *Extension of the Elo rating system to margin of victory*, [DOI 10.1016/j.ijforecast.2020.01.006](https://doi.org/10.1016/j.ijforecast.2020.01.006) | Publisher abstract reports improved tennis forecasts from within-set margin information; among investigated formulations, only joint additive MOV yielded unbiased, stable-variance ratings in simulations. | Consider dominance information, but do not invent an arbitrary margin multiplier and call it a validated method. Full methods were not retrieved in this search. |
| Angelini, Candila and De Angelis (2022), *Weighted Elo rating for tennis match predictions*, [DOI 10.1016/j.ejor.2021.04.011](https://doi.org/10.1016/j.ejor.2021.04.011), [author manuscript](https://cris.unibo.it/bitstream/11585/821483/2/Weighted%20ELO%20rating%20predictions%20in%20tennis.pdf) | Proposes scoreline-weighted Elo and uncertainty estimation; evaluates against standard Elo using tennis histories. | A legitimate additional rating family. Match score is historical input only; the target match's score must never be a predictor. |
| Spanias and Knottenbelt (2013; online 2012), *Predicting the outcomes of tennis matches using a low-level point model*, [DOI 10.1093/imaman/dps010](https://doi.org/10.1093/imaman/dps010), [publisher abstract](https://academic.oup.com/imaman/article-pdf/24/3/311/2666456/dps010.pdf) | Derives game/set/match probabilities from estimated serve/return point probabilities under an IID point assumption. | Supports a mechanistic point-model challenger; IID and point-rate estimation are assumptions. It does not establish that this approach beats Elo here. |
| Buhamra and Groll (2025), *Statistical enhanced learning for modeling and prediction tennis matches at Grand Slam tournaments*, [arXiv:2502.01613](https://arxiv.org/abs/2502.01613), [full text](https://arxiv.org/html/2502.01613v2) | Studies Elo-enhanced regression and random forests; expanding evaluation predicts the 2022 Slams sequentially using earlier tournaments. | Supports domain ratings as model inputs and temporal evaluation. Grand Slam population, updating rules and metrics differ from the present protocol; do not transplant headline accuracy. |
| Poudel et al. (2026), *A Unified Benchmark of Machine Learning and Deep Neural Networks for Tennis Match Prediction*, [DOI 10.3390/analytics5030022](https://doi.org/10.3390/analytics5030022), [full text](https://www.mdpi.com/2813-2203/5/3/22) | Reports Elo–ML accuracy 67.52% and log loss 0.6154, versus Elo 65.87% and 0.6234; larger neural networks showed little accuracy benefit. | Method inspiration only. The reported 133,138 matches / 266,276 player observations do not reconcile with listed split counts totaling 495,374. Exact split dates and independent match counts require reconciliation before any direct benchmark comparison. |
| Arcagni, Candila and Grassi (2023; online 2022), *A new model for predicting the winner in tennis based on the eigenvector centrality*, [DOI 10.1007/s10479-022-04594-7](https://doi.org/10.1007/s10479-022-04594-7), [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8900648/) | Uses recency-weighted loser-to-winner networks to derive Bonacich ratings and logistic probabilities. Men's testing included 4,141 matches in 2016–2020, with periodic refitting, restricted to Masters and Slams. | A credible opponent-network challenger. Different event selection and updating prevent direct score comparisons; every graph must be built at the prediction cutoff. |

## Evidence-informed reopening

**I9 — idea, literature-inspired:** add an opponent-network rating to the shortlist. A graph can propagate new information about a player's past opponents even when that player has not competed. Test a deterministic, cutoff-specific centrality feature before a neural graph model. Compare against Elo on the same rows; reject if its incremental gain disappears with stronger ratings or date-order safeguards.

**I10 — idea, literature-inspired:** use score dominance in a separately identified rating model or point-rate feature block. First verify coverage and reproduce the published update rule. Both Kovalchik's MOV work and WElo motivate the question, but neither licenses using post-match target statistics.

**Working recommendation:** start with I1, I2, I4 and I5, retaining the original eight-subset experiment and overall Elo. I3 is an inexpensive ablation. I6/I9/I10 are worthwhile later challengers; I8 requires stronger evidence of complementary information before spending substantial compute. I7 belongs in a separately labeled information regime if eligible historical odds can be obtained. This is a feasibility recommendation, not a measured model ranking.

## Adversarial review requirements

The originating research agent performed the following self-critique; it does not substitute for independent review by the implementation/audit agents.

| Risk | Observation that would undermine the apparent gain | Mitigation and remaining uncertainty |
| --- | --- | --- |
| Feature leakage | Perturbing target/future results changes earlier feature rows | Test cutoff invariance, eligibility and same-date handling; source tournament-week dates still limit information freshness |
| Repeated validation tuning | Many revisions are guided by the same 2023 errors | Freeze search protocol before validation; log all attempted candidates; keep 2024–2025 sealed |
| Opponent/surface confounding | Raw form gains disappear after strong Elo enters the model | Run grouped removal ablations; describe predictive increment, not causal contribution |
| Sparse histories | Gains depend on dropping new or grass specialists | Retain every target row; shrink estimates and include history counts |
| Orientation artifact | Swapping players does not complement predictions | Train on mirrored differences or explicitly symmetrize inference; test p(A,B)+p(B,A)=1 |
| Calibration overfit | In-sample or same-fold calibration improves while next-year loss worsens | Fit calibrators/blend weights on genuinely held-out temporal predictions; keep calibration inside model selection |
| Misleading uncertainty | Row bootstrap gives overly narrow intervals with repeated players/events | Use paired tournament/week block bootstrap for loss differences; acknowledge remaining cross-event player dependence |
| Algorithm-first optimism | Deep/complex models only improve training fit or one lucky season | Compare to regularized logistic and Elo; report all folds and computational cost |
| Publication or benchmark mismatch | Published results use Slams, later odds, different exclusions or prediction timing | Reimplement accessible competitors on the identical evaluation manifest before any SOTA claim |

## Next experiment and decision record

Decision D1: broaden the experiment under the user's authorization, while preserving the original scientific question and held-out test. Owner: project user; implementing agents choose routine settings within that authorization. Candidate model choice remains provisional until measurements are produced.

Recommended evaluation sequence: freeze the feature/data protocol; generate expanding-fold predictions within 2016–2022; select hyperparameters and any blending/calibration without 2023 labels; train frozen candidates through 2022; produce one 2023 table; choose and serialize the recommended model; hand over hashes, probabilities, reproduction command and uncertainty/ablation outputs for independent final testing. If follow-up searches use 2023 results, label them exploratory and record the deviation.

For a SOTA claim, require a defined task/cohort/cutoff, multiple strong reproduced competitors, an untouched final test, paired uncertainty estimates, and enough evidence that the effect is robust across seasons. A high score on this repository alone supports “best among the models evaluated here,” not a universal state-of-the-art claim.

## Search and provenance log

The search was bounded to publicly accessible primary publications and author manuscripts, emphasizing tennis pre-match prediction, Elo, surface, serve/return and modern models. Search queries included `tennis match prediction Elo surface Kovalchik 2020 2024 prediction probabilities`, `tennis match prediction serve return statistics machine learning 2025 CatBoost log loss`, `Kovalchik 2020 extension Elo professional sport tennis margin victory forecasting International Journal Forecasting`, `tennis predicting outcomes serve return opponent adjusted point based Knottenbelt Spanias Madurska 2012`, and exact-title searches for Weighted Elo and MOV Elo. Citation chasing added opponent-network models.

The academic-search MCP was unavailable. Endpoint preflight reported CrossRef, PubMed and arXiv reachable; OpenAlex fallback discovery returned HTTP 429. Web discovery then retrieved publisher/arXiv/author sources. Direct ScienceDirect retrieval returned 403 for two papers, so those entries rely on indexed publisher abstracts plus the accessible author manuscript where linked. The Bristol 2025 graph-model manuscript could not be retrieved (403), and a Glicko PMC page presented a browser check; neither is used as substantive evidence here. This is not a systematic review and does not establish novelty or exhaustive SOTA coverage. Casual repository claims and incomparable headline accuracies were excluded as evidentiary benchmarks.

Scientific-brainstorming and nature-academic-search skills structured ideation, critique and verification. The procedural skill reference was checked at arXiv (latest record v2, 2026): Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M., *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [DOI 10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065). This is workflow attribution, not evidence for tennis prediction performance.

## Implemented point-feature checks

**Located local evidence, not model-performance evidence:** `scripts/serve_features.py` builds 68 numeric summaries for all 18,774 development targets. Only pinned 2006–2023 annual files were downloaded and SHA-256 verified. All 44,164 eligible Tour history IDs joined to corrected source records; 44,085 had paired valid point statistics and 79 were retained as missing-statistics history. The 2023 364-day overall window has no valid point history for 168 of 5,176 player appearances (165 matches have at least one such player); the same-surface window has 521 such appearances (459 matches). Fixed priors preserve these target matches.

Six focused tests cover inclusive windows, opportunity weighting, paired missingness, invalid-count rejection, player exchange, unseen players, identity mismatches, final-holdout guards and future-statistics poisoning. Command: `.venv/bin/python -m unittest tests.test_serve_features -v`.

A separate direct summation from raw 2022–2023 rows reproduced the 364-day overall summaries for the 2023 Cincinnati final (`2023-422:55`). Alcaraz: 3,420/5,150 service points and 2,349/5,480 return points across 66 valid matches; Djokovic: 2,867/4,057 service points and 1,847/4,421 return points across 51. With the fixed priors, A-minus-B service-rate difference is −0.0415068100851117 and return-rate difference is 0.010960559206379783. These matched the prefix-sum engine to less than 1e−12. Source periods obey T−364 through T−21; statistics from the final itself were excluded.

Remaining interpretation limits: aggregate point rates still reflect opponent mix; opportunity weighting favors longer matches; the fixed priors are pragmatic rather than estimated posterior population distributions; and completed-match eligibility excludes retirements. These are reasons to run the planned incremental ablation, not reasons to change the validated feature definition after seeing 2023 scores.
