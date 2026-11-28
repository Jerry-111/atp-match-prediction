# Round 8: a historical opening-market strength graph

Protocol proposed on 2026-11-27, before implementing or scoring this branch.
This is coordinator proposal P8-4, introduced after the independent Round 8
brainstorm and a literature check. It replaces the deferred multi-bookmaker
source proposal; no restricted tennis-data files or mirrors are used. The
scientific-brainstorming evaluation and responsible-AI workflow applies, with
AI collaborators rather than independent human experts. Prior 2023 results
and the user's 70% objective are acknowledged anchors, not evidence for this
mechanism. See the [round register](round8_protocol.md).

## Question, mechanism and possible failure

Can past opening quotes supply a useful, dynamically updated estimate of
player strength for a future pair, even when that pair's own quote is not
available? Quote edges connect opponents, so a shared latent strength can
transfer information across unplayed pairs. This is a new **historical-market
input, T−21** condition. It differs from Round 7's static market-supervised
student: here eligible past 2023 opening quotes may update inference state.
It also differs from the current-match opening/closing branches: no quote for
the target or any match after its cutoff may enter its prediction.

Possible failure: sparse Tour-only quote coverage, injury or matchup effects
that do not obey an additive graph, decay removing useful links, a single
bookmaker's correlated errors, or a market already dominated by ranking and
Elo. Missing players and disconnected components must not become losses by
assignment. Accuracy gains must survive the full original cohort and loss
guard. Better historical quote reconstruction need not imply better outcomes.

## Primary method and explicit adaptations

Penn, Michael and Bhatt, *Dynamic Graph-Based Forecasts of Bookmakers' Odds in
Professional Tennis*, [author full text](https://arxiv.org/html/2508.15956v1),
published DOI [10.1098/rsos.252512](https://doi.org/10.1098/rsos.252512).
Methods 3.1–3.4 describe match-format conversion, additive log-odds edges,
surface weights, exponential age weights and weighted least squares for player
strength. The paper excludes some tied-rating predictions from its evaluation;
we retain every target. Its reported cohorts cannot establish performance on
this project's Tour season. The publisher returned 403; the author's full
text and public implementation were accessible. No paper dataset was loaded.

Code was checked at author repository commit
`aaea0a02d6b62f37fab2bc2445947b46873dac8b`:
[loader](https://github.com/mpenn114/rss-wimbledon-2025/blob/aaea0a02d6b62f37fab2bc2445947b46873dac8b/src/main_package/data/load_data.py)
and [graph fit](https://github.com/mpenn114/rss-wimbledon-2025/blob/aaea0a02d6b62f37fab2bc2445947b46873dac8b/src/main_package/model/train.py).
The loader uses reciprocal decimal odds despite the article's ambiguous
equation (3). Its negative-logit rating convention is internally consistent.
We use the equivalent positive natural-logit convention below. The code moves
men's BO3 quotes onto a BO5 scale; the article's prose describes BO5-to-BO3.
We choose the latter and convert predictions back to the target format. The
author code adds a 0.001 squared-rating penalty; we retain that fixed value.

This is a **method adaptation**, not a direct reproduction: one permitted
bookmaker rather than the paper's averaged odds, different years and population,
strict T−21 provider-reported-UTC filtering, sparse exact solves rather than numerical
optimization, a conservative identity linkage, no unknown-as-worst assignment,
and full-cohort original-model fallback. Hyperparameters are not taken from
the authors' 2024–2025 selection or results.

## Fixed input and timing contract

- Read only the existing, hash-pinned Valuebetennis 2021, 2022 and 2023 files
  and the audited Round 5 outcome-blind links. Source fields are match/event
  IDs, UTC match time, the two player IDs, and the two opening odds. Never
  read source winner, score, duration or closing odds. Never open a 2024 or
  2025 file. Retain the manifest, link and source-file checksums.
- Restrict graph edges to the already linked Tour targets. This avoids a new
  identity-matching search or unverified cross-level ratings. The original
  targets supply preknown player IDs, surface and the verified best-of-five
  flag. ATP outcomes are used only for historical model selection/scoring,
  never to orient or weight graph edges. Verify one-to-one match IDs, fixed
  identity pairs, events and UTC timestamps against the raw source.
- `p_A=(1/odds_A)/(1/odds_A+1/odds_B)`, requiring both finite odds greater
  than 1. Both sides refer to the fixed A/B identities, never winner/loser.
- For a target tournament date T, use source match UTC timestamps no later
  than midnight UTC at T−21, **inclusive**. This conservative boundary excludes
  quotes later on that boundary day. It is stricter than treating the entire
  date as available. The provider calls quotes prematch but has no capture
  timestamps. Its match UTC is a provider field, not an independently verified
  actual start, finish or quote-availability upper bound. Later metadata checks
  found contradictory own-round times; the post-result clarification below
  records their limits without modifying the frozen numerical experiment.
- Create the graph roster solely from valid edges already available at that
  cutoff. Never use a future appearance to initialize a player. Sort and batch
  equal timestamps deterministically. Same-cutoff queries share identical state.
  No target match statistics or current-match quote availability is an input.
- No usable quote edge exists before 2021. Preserve all 36,421 development
  feature rows and all 2,588 validation rows. Probability files cover only
  2021 H2, 2022 and 2023, where honest R4 OOF or frozen validation predictions
  exist; unavailable predictions use those probabilities exactly. No old
  training row receives a pseudo-OOF probability from a future-fitted model.

## Prespecified small menu

Six structural settings cross the following fixed factors. There is no
additional regularization, calibration or feature grid.

| Factor | Values | Interpretation |
| --- | --- | --- |
| Quote-weight half-life | 90 days, 365 days, infinity | Rapid adaptation, slow adaptation, and a time-flat accumulated-history control |
| Surface weights | all 1; same target surface 1 and other surfaces 0.25 | Common strength versus a partially pooled target-surface graph |

The infinity setting is the **no-decay/static-weight control**, not a frozen
roster: new eligible historical quotes still enter. For clarity about 2023
online updates, additionally report a locked-history diagnostic for the frozen
selected graph, excluding every source timestamp from 2023. This diagnostic
is not eligible for selection and adds no hyperparameter search.

For each graph evaluate its direct probability and a fixed 50/50 arithmetic
mixture with the R4 reference: **12 candidate entries from six graph settings**.
The reference remains eligible, making 13 selectable entries. R5's pooled
exploration and best-loss spline, and Round 7's market student, are report-only
comparators when their existing predictions are available. They do not alter
selection or the reference. The 0.5 classification threshold is unchanged.

Let `B3(q)=3q²−2q³` and `B5(q)=10q³−15q⁴+6q⁵`. Invert the appropriate
monotone polynomial to obtain a set probability q and express every edge on
the BO3 scale `z=logit(B3(q))`. This IID-set approximation and its limitations
are shared with the paper's format rationale. For source edge i and cutoff c:

```
w_i = surface_weight_i * 2 ** (-(c - UTC_i).days_exact / half_life)
min_r  sum_i w_i * ((r_Ai - r_Bi) - z_i)**2 + 0.001 * sum_j r_j**2
```

Use weight 1 for infinite half-life. Solve the sparse Laplacian normal
equations; the fixed ridge removes the constant-shift ambiguity and provides
a weak neutral prior. A target's graph BO3 probability is `expit(r_A-r_B)`;
invert B3 and apply B5 for BO5 targets. The two players must both have past
valid edges and belong to the same connected component. Otherwise the graph
probability and every mixture return the R4 probability **exactly**. No target
is dropped, and a disconnected component's arbitrary level is not compared
with another component. Availability is symmetric; rating/logit differences
are antisymmetric under player exchange.

## Selection and frozen reporting

1. Generate causal predictions chronologically; graph estimation consumes
   only historical prices, with no binary fitted head or learned calibration.
   Select the 13 entries on **2021 H2** accuracy subject to loss ≤ R4 +0.005;
   break ties by loss and then name. All candidates use identical target rows.
2. Evaluate that early choice on 2022. Require accuracy gain ≥0.005 and the
   same loss guard to adopt this branch's candidate. This reused gate is an
   exploratory decision rule, not a significance test.
3. Choose the pooled H2+2022 candidate by the same guarded-accuracy rule. Save
   the menu, scores, early/pooled names, gate and source/code hashes **before**
   computing new 2023 outcome metrics. The gate determines whether to adopt
   the pooled graph candidate or retain R4 in this information condition.
4. Score the full unchanged 2023 cohort once, preserving all candidates,
   the locked-history diagnostic, coverage by year/surface, exact-fallback
   counts, calibration and paired tournament-date block intervals. Intervals
   omit uncertainty from the adaptive multi-round search and shared players.
   Do not revise the menu or selected name after these scores.

The final test remains sealed. Any future extension must preserve fixed
parameters and the information contract, and be separately authorized before
loading new source years. Success or failure here cannot establish SOTA.

## Required verification and artifacts

Synthetic checks cover a hand-solvable graph, weighted cycles, sparse/dense
agreement, BO3/BO5 inversion, player exchange, same-time permutation, future
price poisoning, boundary timestamps, unknown/disconnected fallback, source
direction and source outcomes being ignored. Validate the development-year
guard before parsing future prices or IDs. Rebuild real graph features and
confirm finite outputs and unchanged row/index identity. Report source-link
coverage without using outcomes to modify eligibility.

Planned modules: `scripts/market_graph_features.py` and
`scripts/train_market_graph_round8.py`. Planned outputs under
`results/round8/market_graph/`: graph history and features, temporal comparison,
all predictions/metrics, frozen selection/bundle, source registry and Chinese
report. An independent agent must reload probabilities, verify graph timing
and source columns, reselect the historical gate, and check fallback/metrics.

Coordinator review approved this menu before implementation. A label-free
source inspection found 2,374 / 2,480 / 2,487 linked usable opening edges in
2021 / 2022 / 2023 (7,341 total; 444 players). At the strict cutoff, 2,412 of
the 2,588 targets have both players previously observed; connectivity may
further reduce graph coverage. These are source-coverage counts, not scores.

Implementation note before any graph CV/2023 score: the first source build
stopped because ATP alphanumeric IDs were incorrectly parsed as numeric
provider IDs. The namespaces now remain separate, with the same strict
bidirectional identity check and a regression fixture using real-form ATP
codes. No target, edge eligibility rule or experiment setting changed.

## Post-result timestamp clarification (2026-11-27)

A subsequent chronology investigation found eight same-player/event groups
with different knockout rounds at identical provider UTC and 14 nonincreasing
own-round transitions in the 7,341 frozen edges. The provider field therefore
cannot certify universal actual start/completion time. It is not a quote-capture
or historical publication timestamp. These are metadata checks, not a new
performance-dependent filter.

The [persisted timing audit](../../results/round8/market_graph/timing_audit.json)
checks all 36,421 original development queries: every admitted edge also has
its linked corrected tournament date at or before T−21. There are zero nominal
date violations among 374,716 logical snapshot-edge uses, including all 2,588
2023 targets. An additional dual cutoff would remove no existing edge; no such
algorithm change is made here. This establishes the original nominal date guard,
not independently certified real-time quote/result availability.

The original registry and all numerical/model artifacts remain frozen.
[Reversible text provenance](../../results/round8/market_graph/text_revision.json)
records the exact explanatory changes and reconstructs the at-fit source/protocol
hashes. The [freshness feasibility note](round9_freshness_feasibility.md) links
primary counterexamples and explains why a shorter history lag is not yet approved.
