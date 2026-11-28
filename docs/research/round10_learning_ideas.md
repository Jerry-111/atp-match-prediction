# Round 10 learning ideas — proposals, not fitted results

Initial ideas were sent independently to the coordinator during the Round9
neural run, before this author inspected its 2023 predictions. The previous
nine rounds, the 70% goal and the coordinator's prompts about mechanisms,
objectives and data are acknowledged anchors. The features agent's new ideas
were not consulted. Literature checking followed the initial two proposals.
The scientific-brainstorming workflow preserves rival explanations and the
coordinator's subsequent timestamp objection below. No new model was fitted.

Keep all 2,588 targets, their labels, threshold 0.5 and the 1,812-correct target.
2024–2025 remain sealed. Every further 2023 result is exploratory. The first
idea stays in the original historical T−21 condition. The second would require
a separately named later-information condition and new availability evidence.

## L10-A: conditional expert reliability with honest temporal stacking

**Hypothesis:** different mechanisms fail in predictable, preknown situations.
A point model may be useful when both players have substantial point exposure,
while a supervised feature model may be more reliable with sparse histories.
Earlier global mixtures use the same weight for every match; a low-capacity
gate would change reliability weights by context rather than add another
feature/regularization grid or choose a lucky seed.

**Proposed minimal contrast:** freeze a small primitive expert set representing
different mechanisms before scoring: an existing additive spline classifier,
the R8 jointly fitted point model, and optionally the R9 raw seed-average TD-S
recipe. The exact two-versus-three-expert choice still needs coordinator review;
it must not be chosen from their new 2023 disagreement/error patterns. Compare
an equal-weight mixture, a learned global simplex mixture, and one conditional
simplex gate, all on exactly the same expert forecasts and meta-training rows.
The existing R4 recommendation is a reference, not automatically a safe meta input.

For the conditional model, use at most five prespecified **even** inputs such
as absolute log-rank gap, minimum recent point exposure, minimum career count,
same-surface exposure and match format. Learn a regularized linear softmax
gate `g_k(z)` and predict `sum_k g_k(z) p_k`. Since `z` is unchanged by A/B
exchange and each expert complements, the mixture complements as well. Fit
Bernoulli log loss, with one fixed shrinkage toward the fitted global mixture;
do not add player-specific gates, neural routing or dozens of context splits.
Constant contexts must reduce to the global-weight control. This is a frozen-
expert conditional stacking adaptation, not reproduction of jointly trained
mixture-of-experts networks.

**Chronology is the key design constraint.** For an outer year Y:

1. Build primitive expert forecasts for earlier meta-training years, initially
   2017…Y−1. For every such year s, fit that expert using labels strictly before
   s with the 21-day purge; neural epoch selection also stays inside that older
   training period. As-of latent point updates retain their own T−21 rule.
2. Fit global weights and the conditional gate only on those earlier-year
   forecasts and labels. Predict Y using experts fit on eligible pre-Y labels,
   and the already fitted meta models. Never fit the gate on Y's own labels.
3. Repeat independently for 2020, 2021 and 2022. Choose the small procedure on
   early folds and apply the existing reused-2022 practical gate. Freeze it,
   then fit the final meta model on honest forecasts through 2022 for 2023.

**Forbidden shortcut:** a mixture whose weights were selected using all
2020–2022 OOF labels is not an independent OOF feature for a new meta learner
on those same years. Either use primitive, honestly refitted forecasts, or
rebuild *both* the mixture weights and its underlying predictors entirely
inside each earlier period. Likewise, full-data fitted-on-own-label expert
probabilities cannot be renamed OOF. Existing caches are reused only after
checking these exact dependencies. Previously selected fixed recipe constants
remain part of the acknowledged adaptive research history; this does not turn
the reused outer years into a pristine confirmatory test.

**Discriminating result:** conditional gating must beat its identically trained
global-mixture control on later folds, not merely beat one weaker expert. Also
report gate weight variability and performance on prespecified coverage strata,
while scoring the full population. If weights collapse to constants, swap wildly
across forward folds, or improve only an in-sample fit, reject the routing claim.
An error-overlap/oracle calculation on development data can describe a ceiling,
but is not a realizable predictor or a selection metric. Accuracy selection
retains the +0.005 loss guard and fixed threshold; a loss-only gain is reported
as such. The experts may simply make the same errors, leaving little room.

**Cost:** three small meta procedures per outer/final period are negligible.
The actual cost is generating previously unavailable honest expert histories.
Existing joint-point as-of forecasts start in 2017, so a two-expert spline/point
pilot should be minutes to tens of minutes on four CPU threads. Adding earlier
TD-S forecasts could cost roughly another Round9-sized run; do not assume
cached composite forecasts make that cost disappear. Bound this before review.

**Primary evidence, checked 2026-11-28:**

- Jacobs, Jordan, Nowlan and Hinton (1991),
  [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf),
  DOI 10.1162/neco.1991.3.1.79. The author-hosted original was inspected, including
  pp.80–81, equations (1.1)–(1.2) and Figure 1's normalized gate. Its experiments
  concern vowel recognition and jointly trained experts, not chronological
  tennis or frozen forecast stacking. It motivates the mechanism, not a claimed gain.
- Van der Laan, Polley and Hubbard (2007),
  [Super Learner](https://pubmed.ncbi.nlm.nih.gov/17910531/),
  DOI 10.2202/1544-6115.1309. The primary article's abstract explicitly bases
  combination weights on cross-validation. The linked UC item currently resolves
  to Polley's later dissertation, so it is not represented here as the original
  full text. Standard V-fold theory is not automatically a guarantee under our
  dependent player histories and forward-time adaptation.
- Kovalchik (2016),
  [Searching for the GOAT of tennis win prediction](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2015-0059/html),
  §4/Table 5, finds substantial differences across ranking/tournament/surface
  groups. That motivates context as a hypothesis, but does not show learnable
  complementary model errors. Our repeated negative fixed-mixture experiments
  are a further reason to keep the gate small.

## L10-B: completion-certified recent history, separate information condition

**Initial idea:** the unavailable 1–20-day interval may hide useful recent form
or workload. Test actual newly available past results across tournaments, not
only each player's own earlier rounds in the target event (already studied in
R7/R8). This changes information availability and must never be ranked as an
improvement under the original T−21 condition.

**Adversarial correction:** the coordinator's new Valuebetennis audit found
eight player/event UTC groups shared by different rounds and fourteen
non-increasing round transitions. A provider's `Date`/UTC field or a claim that
it denotes match time is not completion evidence. No existing field is assumed
adequate here. The initially suggested date-only shortcut is rejected.

Before fitting, require a new independently checkable completion source: for
example, official archived completed-event records/final draws with established
actual end dates, not merely scheduled dates. At a target's tournament-start
cutoff, permit only certified already-completed events. This is a conservative
alternative to claiming intra-day match availability. Postponements, ongoing
multiweek events, same-day ambiguity and identity conflicts must remain excluded.
If actual match-day forecasting is proposed instead, define and verify that
new decision time separately; do not quietly replace the original target cutoff.
Matching must use preknown identity/event/round fields, not target outcomes.

**Matched pilot, only after source qualification:** hold model, features,
target cohort and scoring fixed; compare existing T−21 history against history
plus only certified new records. An unchanged T−21 rerun is the control. Every
uncertified target/history component falls back exactly to T−21; publish coverage
and availability evidence, never drop difficult matches. Select/freeze using
earlier folds, preserve 2023 as exploratory, and keep 2024–2025 unopened.
No new data download or empirical fit is authorized by this idea note alone.

**Falsification/cost:** if most candidate updates lack completion evidence,
stop at a source-feasibility negative result. If certified updates add no forward
gain over the matched cutoff control, reject the stale-information explanation.
The main cost is provenance: several hundred event records across development
years may be necessary. Model fitting itself can stay within a few existing
small linear recipes. Prefer idea A until a credible completion source exists.

**Located evidence and limits:** Kovalchik's same primary paper, §3.1, updates
ratings through the validation match date; §5 emphasizes dynamic reassessment.
It does not isolate the value of the last 21 days or prove source completion
times. Its 2014 cohort has 2,395 matches, excludes retirements/walkovers and has
68.1% higher-ranked winners, so its 70% Elo result is not a comparable target
achievement for our 2023 cohort. The coordinator's physical timestamp audit is
direct contrary evidence against implementing this idea with existing UTC data.

Decision proposal: review the small honest conditional-stacking design first;
keep completion-certified updates as a source-dependent research question.
Neither proposal is a promised improvement or a new fitted result.
