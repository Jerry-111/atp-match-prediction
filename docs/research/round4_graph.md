# Round 4: paper-informed temporal opponent networks

Date: 2026-11-25, America/Los_Angeles. Decision owner: project user. Implementation and bounded source verification: Codex feature agent. This AI-assisted experiment follows the scientific-brainstorming independent-proposal and adversarial-review workflow and the nature-academic-search citation-verification workflow. It does not claim SOTA or a new scientific discovery.

## Independent hypothesis before reading the paper

**Idea:** indirect opponent quality may complement sequential ratings: defeating a player who defeated strong opponents can carry information beyond a local Elo update. Network-wide reassessment could also change a player's strength estimate when that player has not recently played. **Assumption:** the observed match network contains enough comparable and connected evidence for indirect paths to be informative. **Prediction:** the added graph family improves training-period forward log loss conditional on the existing ratings and statistics. **Falsifier:** the increment is absent, inconsistent across years, or concentrated in sparse/disconnected players whose imputation drives the result.

Selection criteria were fixed before computation: retain the same target rows and T−21 cutoff, guarantee player-exchange symmetry, make the relationship to a verified primary method explicit, limit the feature family to at most 16 columns and build it within 60 seconds. Exact paper formula recovery took priority over trying a broad graph hyperparameter grid.

## Verified primary source and mathematical core

Arcagni, A., Candila, V., and Grassi, R. (2023), *A new model for predicting the winner in tennis based on the eigenvector centrality*, **Annals of Operations Research 325**, 615–632. DOI [10.1007/s10479-022-04594-7](https://doi.org/10.1007/s10479-022-04594-7). Crossref verifies the title, three authors, journal/pages and June 2023 print issue; online publication was 7 March 2022. Citation status: **verified**, with online/issue years distinguished.

The paper directs edges from losers to winners. Equations (1)–(2) assign each result age-dependent weight `1/(1 + age/alpha)`; its application uses `alpha = 365 days`. The rating is the principal incoming eigenvector: `W.T @ c = lambda_max * c`, normalized to Euclidean norm one. Appendix Eq. (11) describes the eigenvector centrality; this is not Katz centrality or PageRank. Its probabilities come from a fitted logistic model. [Primary full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8900648/).

The original men's sample is restricted to Masters/Grand Slams and uses a different forecasting/refitting protocol. Its published accuracy or loss cannot be treated as directly comparable to this repository's broader target cohort and prediction cutoff.

## What is reproduced and what is adapted

**Reproduced mathematical core:** loser-row/winner-column weighted adjacency; repeated results add edge weight; hyperbolic aging with the paper's single fixed 365-day parameter; the Perron eigenvector of the transpose; and L2 normalization. The pure `bonacich_scores(..., regularization=0)` helper computes that core for graphs with a usable positive principal eigenpair. A hand-check uses `W=[[0,4],[1,0]]`, giving centralities `[1,2]/sqrt(5)` and correctly favoring the player with four wins.

**Production adaptations, fixed before scoring:**

1. Historical evidence is the project's eligible completed, ranked Tour/major main-draw inventory from 2006–2023. Target seasons are 2009–2023. No test outcome is parsed into the graph.
2. Each unique target date T queries all results through T−21 inclusive. Hyperbolic ages are measured relative to that information cutoff. Matches with the same date enter together, and every target sharing the date receives the same graph snapshot. No match-result ordering within a tournament is invented.
3. The vertex set contains only players observed in eligible history by the cutoff. It does not include players known only from future rows or upcoming target matches. This differs from a fixed full-sample vertex roster and prevents future identity/count information from changing normalization or regularization.
4. Sparse directed graphs can be reducible. Production therefore computes the Perron vector of `W.T + epsilon/n * 11.T`, with `epsilon = 0.01 * sum(W)/n`. This uniform positive pseudolink term is a declared regularization adaptation, not an equation attributed to the paper. It makes every observed node reachable without constructing a dense matrix. There is no row normalization, so this remains regularized incoming eigenvector centrality, not a random-walk score.
5. A player absent from a snapshot receives the mean centrality of its observed roster. Two unseen players have a zero difference. Missing players do not change the graph. Log-score features add `0.05 * mean_centrality` to both scores; this prevents arbitrary infinite ratios and is a fixed prior rather than a fitted outcome-dependent parameter. Missing-player indicators remain available to the learner. Empty graphs give neutral differences.
6. The same computation is applied once to overall history and separately to the current surface. The surface-specific graph is an inspired extension, not part of an exact reproduction of the original empirical model. Both use the same alpha and pseudolink settings.
7. A downstream project model receives A-minus-B score differences with exchangeability enforced. It does not reproduce the source paper's full logistic specification, population, temporal refitting schedule or reported empirical results.

Only **one memory setting** is used, with two graph scopes. No new 2023 score was inspected or computed by this feature implementation to choose parameters.

## Output and use

`build_paper_graph_features(targets, inventory)` in `scripts/paper_graph_features.py` returns **12 numeric columns**, preserving input target index and order. Each graph scope contributes B-score difference, smoothed log B-score ratio, log weighted-win and weighted-loss differences, and observed-player difference/sum. Every `_diff` field reverses sign when A and B are exchanged; other fields are symmetric. No target label, ID encoding, future match result or future roster count enters the numeric output.

The sparse solver has a deterministic uniform start and verifies its eigenpair residual. It raises on an unusable solution rather than silently substituting a performance score. A graph with no edges returns uniform centrality. Production pseudolinks guarantee a positive principal eigenvector for nonempty observed graphs.

The family can be appended to the existing 202-column Round 3 matrix and evaluated under the parent's training-only selection protocol. The original B-score and the invented regularization/surface/log transformations must remain distinguished in reports. Do not label the resulting candidate a direct replication of the paper's benchmark.

## Adversarial checks and remaining uncertainty

- Indirect paths can mostly encode match volume or repeated elite-player encounters. Weighted win/loss strengths are exported as simple controls; the graph should demonstrate predictive increment over existing ratings rather than merely a high marginal association.
- Centrality has population-dependent scale despite L2 normalization. The roster, missing-player prior and surface graph composition can shift. The log ratio and explicit observed flags help represent this but do not establish calibration.
- Hyperbolic memory has a long tail. Players with extensive old histories may retain influence, even after inactivity. The single paper-based alpha is a hypothesis, not an optimized tennis truth.
- Pseudolinks and average-player newcomer priors improve numerical coverage but can distort sparse grass or new-player comparisons. Preserve those observations and inspect subgroup errors without retroactively selecting parameters from 2023.
- Multiple previous rounds have already inspected 2022 and 2023. Any new reported scores on those years are exploratory. Graph-feature correctness cannot repair repeated-model-search optimism; the final test remains unopened.

Decision: advance this bounded, documented family to the parent's declared comparison. Retain null and harmful results. Revisit parameters only as a separately logged exploratory experiment using training-period evidence.

## Verification and retrieval log

Command: `.venv/bin/python -m unittest tests.test_paper_graph_features -v`.

Six tests cover the exact two-player calculation; indirect credit and positive regularized centrality; vertex permutation; inclusive cutoff; poisoning later outcomes and entirely new future player IDs; held-out sentinel labels; same-date row permutation; surface separation; A/B exchange; finite unseen-player priors; duplicate/invalid input rejection; and held-out target locking.

Full expanded cohort benchmark: **36,421 × 12 finite features in 10.308 seconds**, including loading the published inventory and expanded target CSV in the local Python 3.12 environment. This is implementation/runtime evidence only, not model-performance evidence.

The academic-search MCP was absent. The known DOI was verified directly against Crossref. The PMC primary page initially returned full text; subsequent requests encountered its browser challenge. Official NCBI BioC supplied the full article prose and appendix; Europe PMC full-text XML supplied MathML for Eqs. (1)–(2). Follow-up XML requests for Eq. (11) timed out, so its principal-eigenvector and Euclidean-normalization definition was cross-checked in the retrieved appendix prose rather than inventing a missing formula. The publisher DOI redirect was inaccessible through web retrieval. This is one-paper method verification, not an exhaustive graph-model literature search.

Skill workflow attribution is already recorded in [the project evidence register](sota_evidence.md); the primary method attribution above is separate from that procedural attribution.
