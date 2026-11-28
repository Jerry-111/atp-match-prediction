# Round 8: historical tournament environment × player style

Recorded 2026-11-27 before this branch is fitted. The complete 2,588-match
2023 cohort, binary labels and threshold 0.5 stay fixed. All 2023 results are
exploratory after earlier inspections; 2024–2025 remain sealed.

## Hypothesis, provenance and evidence

The audit agent independently proposed A8-2 before reading this round's other
agents' proposals: broad clay/grass/hard categories may conceal tournament
conditions that favor different player styles. An independent features-agent
proposal overlaps, but agreement is not evidence. Known earlier indoor-metadata
and joint-point-model proposals are disclosed as possible anchors. This is
AI-assisted use of scientific-brainstorming, followed by primary-source checks.

Sackmann's [2013 analysis](https://www.tennisabstract.com/blog/2013/11/19/the-speed-of-every-2013-surface/)
uses server/returner-adjusted ace frequency as a playing-environment proxy and
warns about small-event noise. The [2012 primary table](https://tennisabstract.com/reports/surfaceSpeedATP.html)
documents 3/2/1 weighting across three years. The proxy combines court, balls,
altitude and weather; it is not a measured physical Court Pace Index. We do not
copy the published retrospective season rates into forecasts. All player
adjustments below use earlier observations only.

Ingram's [2019 primary paper](https://martiningram.github.io/papers/bayes_point_based.pdf)
includes tournament effects in a joint serve/return model, but its best reported
match result is worse than its Elo comparator. Gollub's [2021 primary paper](https://journals.sagepub.com/doi/pdf/10.3233/JSA-200345)
also treats limited sample sizes and schedule strength as problems for serve
forecasts. These sources motivate controls; none proves this branch will reach
70%. Our feature-based interaction model is an adaptation, not a replication
of either hierarchical model. This is a focused search, not a systematic review.

**Falsifying outcomes:** forward-year performance does not improve over the
same model with surface-only context; raw and adjusted environment perform
similarly; gains depend on incorrectly pooled relocated events; or results are
explained by generic strength interactions rather than style interactions.
Tournament participation, statistical coverage and temporal drift remain rival
explanations. No association is interpreted as a causal effect of court speed.

## Identity and time rules

The venue proxy key is `(tournament-ID suffix, normalized tournament name,
surface)`. Normalization only removes case, accents and punctuation. Different
names are not automatically merged. The metadata scan of 2006–2022 found 114
ID suffixes, 20 with multiple names or surfaces: suffix 424 includes San Jose,
Memphis, New York and Dallas. Therefore suffix alone is unsuitable. This
conservative key still cannot detect all within-city venue changes; the feature
is named historical tournament environment, not a verified physical venue.

Only the preceding three **seasons** contribute to a query, with weights
3/2/1, and every contributing date must be at most target T−21. No current
edition or current-season statistics enter the environment. In production the
raw loader opens only 2006–2022 for 2009–2023 targets; target-year metadata come
from the pinned inventory. Source eligibility follows the existing cleaned Tour
inventory, preserving its exclusions and duplicate handling. No new targets are
added or removed.

For each historical match, server, returner and tour rates are queried from
that match's T−364 through T−21 window. Thus even the adjustment of an old
match cannot use its outcome, same-event points, later results or full-season
averages. These rates have fixed 100-point priors: 0.65 for service-point wins
and 0.06 for aces. The returner's allowed rate uses the opponents' service
opportunities. Our explicit expected-rate adaptation is

`expected = sigmoid(logit(server_rate) + logit(returner_allowed_rate) - logit(tour_rate))`.

Sum observed and expected successful points over both servers. Compute
smoothed observed-versus-expected log-odds shifts with a fixed 2,000-point
prior; subtract the corresponding surface-wide shift. This isolates a
tournament deviation from the broad surface as far as these proxies allow.
Raw observed tournament-versus-surface shifts provide a matched control for
player adjustment. Surface-versus-tour observed shifts provide a broad context
control. A missing tournament history produces zero tournament-deviation
features and an explicit flag; every target remains. This is neutral feature
fallback, not a claim of exact fallback to an old model's probability.

## Fixed feature interactions and model comparison

The old 326 historical features stay bit-for-bit unchanged. The two context
signals are serve-point and ace log-odds shifts. Six existing odd style fields
are used: ace rate, first-serve points won, second-serve points won, first-serve
return points won, second-serve return points won, and height. Multiply each
context by each style difference. Products remain odd under player exchange.
An additional control multiplies adjusted tournament context by the old
log-rank and blended-Elo differences. Merely adding a shared intercept does
not establish a mechanism for changing the favored player.

Ten fixed fits: five feature groups × linear or existing odd-spline learner.
All use C=0.01, three-year recency half-life and the same eligible 2009–2022
training targets. Groups are:

1. Original 326 inputs.
2. Original plus surface-context × style.
3. Group 2 plus raw tournament-context × style.
4. Group 2 plus adjusted tournament-context × style.
5. Group 2 plus adjusted tournament-context × rank/Elo.

There is no C, prior-strength, lookback or interaction search. Controls keep
all missing-history targets. Any learner transform fits only its chronological
training subset. Exact A/B probability complementation is required.

Selection is fixed before fitting: ten new configurations plus the R4 reference
are the only eligible single models; no simplex, learned mixture, or fixed
blend is added. Select on pooled 2020–2021 forward predictions by accuracy,
then log loss, then name, subject to log loss no more than R4 +0.005. Evaluate
that choice on 2022; adoption needs at least +0.005 accuracy and the same loss
guard. Only if that gate passes may the independently pooled 2020–2022 choice
replace R4 for this branch. Every fold uses earlier seasons with a 21-day purge
against its earliest validation date. Freeze selections before 2023 scoring.
All ten are finally fitted on eligible 2009–2022 rows for matched comparisons.
R5 best-accuracy and best-loss models are report-only controls.
All group-by-learner contrasts and per-year results, including negative ones,
will be saved. A repeated 2022 gate cannot count as independent confirmation.

## Audit and interpretation

Tests must cover source-year filtering before field access, current-edition
poisoning, stable identity under year/name changes, deterministic prior
adjustments, invalid counts, no target-label dependence, and full A/B exchange.
The source/feature/menu hashes and frozen choice precede new 2023 scoring.
Report coverage with full-cohort results; sparse-event subgroups cannot replace
the denominator or be used to manufacture 70%.
