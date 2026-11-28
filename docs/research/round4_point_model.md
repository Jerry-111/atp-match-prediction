# Round4: paper-supported scoring aggregation

Date: 2026-11-25. Owner: audit/point-model agent. This is an AI-assisted research
proposal and implementation; no new2023 performance was used to design it.
Earlier2023 results were already visible, so the broader round remains exploratory.

## Independent hypothesis and falsifier

**Idea P4-1.** Convert the existing opponent-adjusted service-point probabilities
into game, set and match probabilities, rather than asking a generic learner to
infer every consequence of tennis scoring. The best-of-five format should amplify
a stable point-level advantage more than best-of-three. The existing inputs
already contain IID service-game probabilities; the new information is their
aggregation through sets and matches with alternating servers.

**Assumption.** Conditional on server, points are independent with constant win
probabilities throughout the hypothetical match. **Prediction.** The new logit
features could improve forward-fold probability calibration. **Falsifier.** No
gain in the prespecified training-period comparison, or gain restricted to one
season, would favor keeping the existing recommendation. This is a candidate
representation, not evidence of improvement.

## Primary evidence checked

The `nature-academic-search` multi-source-search workflow was used for a bounded
formula/citation check. Academic-search MCP tools were absent; Crossref metadata,
publisher pages, the author's publication list and author-hosted material were
checked instead. Search phrases included the Spanias–Knottenbelt title and
“O'Malley probability formulas winning tennis game set match”. This was not a
systematic literature review.

| Source | What was verified | Access and use |
|---|---|---|
| Spanias, D. & Knottenbelt, W.J. (2013). *Predicting the outcomes of tennis matches using a low-level point model*. IMA Journal of Management Mathematics24(3),311–320. [DOI10.1093/imaman/dps010](https://doi.org/10.1093/imaman/dps010) | Title, authors, journal issue and hierarchical point-model scope; online publication was2012. | Crossref metadata and [author publication list](https://www.doc.ic.ac.uk/~wjk/publications.html) agree. Publisher PDF redirected to an inaccessible endpoint; this implementation does not claim to reproduce that paper's full estimator. |
| Spanias, D. (2015). *Professional Tennis: Quantitative Models and Ranking Algorithms*. Imperial College London PhD thesis. [Author-hosted full text](https://www.doc.ic.ac.uk/~wjk/publications/spanias-2015.pdf) | Chapter3 printed pages60–66, equations3.1–3.7: match/set/game/tiebreak recurrences and serving order. | Full thesis downloaded and text inspected. This is the directly checked source for the implemented recursion. |
| O'Malley, A.J. (2008). *Probability Formulas and Statistical Analysis in Tennis*. JQAS4(2). [DOI10.2202/1559-0410.1100](https://doi.org/10.2202/1559-0410.1100) | Citation metadata and the probability-hierarchy topic. | The author's [2007 NESSIS slides](https://nessis.org/nessis07/James_OMalley.pdf), slide4, give the game formula; slide18 gives a numerical match example used below. The slides are primary presentation material, distinct from the journal article. |

The thesis specifies alternating service games and tiebreak service sequence
ABBAABBA…, with an infinite tied-score tail solved algebraically. It also explains
that the first tiebreak receiver serves first in the next set. Those properties
are preserved in the implementation. The NESSIS example uses service probability
0.65 and return probability0.37 and reports best-of-three match probability0.5985;
our output is0.59853154849. These checks validate arithmetic under the stated
model, not tennis forecasting performance.

## Implementation and explicit deviations

`scripts/paper_point_model.py` implements a vectorized probability calculation.
For service probability `p` and `q=1−p`, the advantage-game probability is
`p^4(1+4q+10q²) + 20p³q³ p²/(p²+q²)`. The tiebreak recursion uses each point's
scheduled server; at6–6 its absorption probability is
`pA(1−pB) / [pA(1−pB)+(1−pA)pB]`.

Our set transition kernel keeps both the set winner and the next server. The
match calculation propagates that joint state until two or three sets are won.
Unknown first server is averaged equally between A and B. This avoids discarding
server continuity while aggregating set results.

**Standardized-rule approximation:** every set uses advantage games and a
seven-point, win-by-two tiebreak at6–6. Historical advantage final sets and later
ten-point deciding tiebreaks are not reconstructed. Consequently, the algorithm is
exact for the stated standardized IID model, not for every historical event's
rules. Best-of-three/five is taken from the existing verified pre-match format
flag. No observed match score, current-match points or outcome enters the input.

The point estimates are the project's earlier approximate opponent-adjusted
dynamic skills, reconstructed from their sum/difference columns for365/730-day
half-lives and overall/surface scopes. They are **not** Spanias and Knottenbelt's
first-/second-serve point-estimation procedure. This contribution is therefore a
paper-supported scoring transformation, not a claimed replication of a published
forecasting system.

For each of the four point-estimate pairs, the builder emits match, set and
tiebreak logits:12 columns named
`paperpoint_h{365|730}_{overall|surface}_{match|set|tiebreak}_logit_diff`.
All change sign when players exchange. Probability clipping at1e−8 is only for
finite logits. No coefficients, thresholds or calibration parameters are fitted
here. The caller owns chronological model selection.

## Independent checks and remaining risks

Tests check the primary worked example, neutrality, player exchange, monotonicity,
best-of format amplification, near-endpoint behavior, missing/invalid input guards,
outcome poisoning and the final-holdout guard. A separate forward point-mass
calculation approximates the full tiebreak tail until remaining mass is below
1e−15. A scalar game-path enumeration retains the next-set server and independently
reconstructs set/match probabilities. Both agree with the production vectorized
calculation to numerical precision.

The main risks are misspecified point estimates, within-match dependence,
nonconstant ability and standardized final-set rules. Score aggregation can
amplify biased point estimates into overconfident predictions; adding these
features is not evidence that raw model probabilities are calibrated. The
prespecified forward comparison and untouched final test must decide whether
this representation helps. Negative results must remain in the record.
