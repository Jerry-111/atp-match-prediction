# Round 6: supervised lower-level transfer

Date: 2026-11-27. This protocol is recorded before fitting the new models. The user wants progress toward 70% on the same full 2,588-match 2023 cohort; 2024–2025 remain sealed. Prior 2023 results have been viewed repeatedly, so every new result is exploratory. The decision owner is the user; this is AI-assisted ideation and implementation, not a human expert consensus. The scientific-brainstorming skill and evaluation/facilitation references were used; attribution remains in [the evidence register](sota_evidence.md).

## Independent hypotheses and evidence check

The research agent formed three proposals before seeing the coordinating agent's suggestions: (R1) use eligible Challenger/qualifying matches as supervised training targets, not only historical rating updates; (R2) estimate lagged event-specific serve conditions and interact them with player style; (R3) fit a strongly regularized nontransitive player matchup representation. The coordinator's separate suggestion arrived while these already-formed ideas were being written; the initial list was not changed in response.

R1 assumes conditional rank/strength relationships transfer across levels. Its falsifier is worse Tour forward accuracy/loss than the identical compact Tour-only model. More source data cannot ensure an improvement when competitive populations, missing information and selection differ. R2 could add omitted venue information, but a lagged proxy may instead measure entrants or obsolete venues. R3 could describe persistent matchup styles, but sparse pairs and changing players make it vulnerable to overfitting. Criteria were information increment, temporal validity, bounded compute and usefulness of a null result; no numeric consensus score was used.

Bounded primary-source checks on 2026-11-27:

- Ben-David et al., *A theory of learning from different domains*, Machine Learning 79:151–175 (2010), [DOI/full primary page](https://link.springer.com/article/10.1007/s10994-009-5152-4). It motivates combining source and target errors while explicitly tying successful transfer to domain divergence and a hypothesis that works in both domains. This is general methodological support, **not evidence that Challenger supervision improves ATP Tour forecasts**; no directly validating primary tennis transfer study was located in the bounded search.
- Gallagher, Frisoli and Luby, *Opening up the court*, JQAS (2021), DOI [10.1515/jqas-2019-0015](https://doi.org/10.1515/jqas-2019-0015), [primary manuscript](https://www.stat.cmu.edu/~brian/valerie/617-2022/617-2021/0%20-%20a%20couple%20of%20IDMRAD-like%20papers/Opening-up-the-court_JQAS_2021.pdf). Its mixed-effects analysis finds player/tournament differences among Grand Slams, but does not validate this proposed past-only venue proxy or a 70% forecast. R2 is deferred for source/venue-continuity work.
- Chen and Joachims, *Modeling Intransitivity in Matchup and Comparison Data*, WSDM 2016, DOI [10.1145/2835776.2835787](https://doi.org/10.1145/2835776.2835787), [author PDF](https://www.cs.cornell.edu/~tj/publications/chen_joachims_16a.pdf). Equation (4) supports an antisymmetric inner-product model. Crucially, §4.4 reports **no test-accuracy improvement** on 23,806 ATP matches, despite a small likelihood gain. This direct negative evidence lowers R3's priority; it is deferred.

Search terms: `tennis prediction Challenger ATP transfer learning train challenger matches paper`, `tennis tournament court speed serve return hierarchical model venue effect paper`, `Chen Joachims 2016 modeling intransitivity matchup comparison data blade chest paper pdf`, and the Ben-David title/authors. Search limitations: focused web retrieval, not a systematic review; no novelty claim. Decision: implement R1 as a small, matched data-transfer experiment.

## Population and information

Use the existing 36,421 Tour development targets (33,833 through 2022 plus the original 2,588 in 2023), keeping their identity, order, ranks and labels. Add **55,969 Challenger and 19,292 qualifying** targets from 2009–2022: 75,261 additional supervision rows and 111,682 combined feature queries. Auxiliary 2023 targets are not fitted or evaluated. Annual source membership, canonical A/B orientation, original IDs, positive pre-match ranks and corrected dates come from the published inventory.

Histories may include eligible Tour/Challenger/qualifying results from 2006–2023, but every query sees only T−21 inclusive, with the existing date-batched multilevel rating engine. No 2024–2025 outcome is inspected. All evaluated outcomes remain Tour outcomes on the original complete cohort. Qualifying and main-draw repeats remain distinct original match IDs.

The shared compact matrix has 48 columns: the existing 37 multilevel features, log-rank advantage, reciprocal-rank advantage, mean log rank, absolute log-rank gap, three surface indicators, and four fixed clay/grass interactions with log-rank and common Elo advantage. Those interactions are odd strength times even known surface. Target scores/statistics never enter predictors. Auxiliary best-of format is not guessed; format is omitted from this shared compact experiment.

## Frozen fitting grid and weight normalization

Twelve fit specifications: auxiliary per-row weight in `{0,0.1,0.3,1}` crossed with logistic `C ∈ {0.01,0.1}` and an odd-spline logistic head with `C=0.01`. All have a fixed three-year recency half-life, no intercept, train-only scales and exact player-exchange symmetry. Zero auxiliary weight removes those rows **before** fitting scales or bases, making it a genuine matched Tour-only control.

Before normalization, a row has weight `2^(-(latest_training_season-season)/3)` multiplied by 1 for Tour and the fixed auxiliary weight otherwise. Normalize the **global total to the number of Tour fitting targets** in that fold. Thus adding auxiliary supervision does not merely increase total weighted sample mass and weaken the effective L2 penalty. It still redistributes emphasis away from Tour, which is the transfer hypothesis being tested. Scaling and basis normalizations use only positive-weight training rows with these weights.

The odd spline uses four fixed strength fields (log-rank, common overall Elo, common surface blend and level-aware overall Elo), quadratic bases with fixed knots at −3,−1.5,0,1.5,3 training RMS units, constant extrapolation, and fixed shape-block weight 0.3. It is a project learner, not a replication of the domain-adaptation paper's bound or estimator.

Every fold uses only earlier seasons and dates through earliest Tour validation date−21 days. Auxiliary matches in the validation year are never fitting targets. The existing lagged history can still incorporate earlier available matches in that year, just as in prior experiments.

## Selection, comparators and freezing

The 12 fitted models have 36 deterministic probability mixtures with the frozen Round 4 reference: new-model fractions `{0.25,0.5,0.75}`. Together with the unblended models and Round 4, there are **49 selection entries**, requiring only 12 distinct model fits per fold. No general weight optimizer or larger mixture search is added.

Select on 2020–2021 by highest accuracy, then lower log loss, then stable name, subject to reference log loss+0.005. The early-selected candidate must improve 2022 accuracy by at least 0.005 and worsen loss by no more than 0.005 against the **current frozen Round 4 model's OOF predictions**. Otherwise retain Round 4. Pooled 2020–2022 selection is frozen before any new 2023 prediction; if the gate fails, it is an exploratory comparator only.

The Round 5 pooled mixture is reported as a comparator, but is excluded from this selection/gate because its mixture weights used all prior pooled folds. Old models and their predictions are unchanged. Report every compact setting and its mixtures, plus all weight-positive versus matched weight-zero differences in CV. The small compact representation may be weaker than the rich reference; that is why both the matched data-transfer comparison and the rich reference comparison are necessary.

Reproducible inference saves compact fitted models plus the trusted prior bundle. Full Tour inference features combine the existing Round 5 matrix with the additional compact fields, checking overlapping historical values. No refitting occurs during artifact reload. All feature/target/source hashes, candidates, weights, fold counts, selected names and final-test flags are preserved in this branch's artifacts.

## Fixed risks and stopping interpretation

Auxiliary labels can transfer badly because rank ranges, incentives, qualification selection and data coverage differ. Large auxiliary weights can dilute useful Tour relationships even after normalizing total mass. Jointly using Challenger and qualifying is a bounded first experiment, not evidence they share the same conditional law. The severe T−21 lag also remains; no model can reconstruct unobserved current fitness from future outcomes. An improvement confined to auxiliary training or a changed target subset is not success.

The 2022 gate has itself been reused across rounds, and the current Round 4 reference was selected using pooled historical folds. This is an exploratory stability rule, not a fresh independent confirmation or significance test. Date-block intervals do not account for all search or repeated-player dependence. No outcome is guaranteed and no failed setting will be hidden.
