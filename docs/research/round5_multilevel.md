# Round 5: cross-level history, fixed prior controls

Status: feature implementation and leakage tests; **no new model scores were
computed to choose these features**. The user requested continued exploration
toward 70% accuracy on the same 2,588 matches. This is a goal, not a supported
forecast. The 2024–2025 final test remains sealed.

## Independent proposal and decision

Following the project's scientific-brainstorming workflow, this agent proposed
three directions before seeing other agents' ideas: F1 additional Challenger
and qualifying history; F2 historical retirement/long-term load; F3 technical
style and matchup interactions. The parent selected F1, then authorized a
bounded F3 branch before any new 2023 score. F2 remains unimplemented: T−21
hides the immediate load period, retirement is not a medical diagnosis, and
completion-filtered targets can limit its relevance. F3 is separately recorded
in [round5_style.md](round5_style.md); indoor/environment estimates are deferred.

**Hypothesis:** players with little Tour history have informative results in
other circuits, and common opponents allow those results to improve latent
strength estimates. This adds actual historical observations rather than
another transformation of the same Tour results. The falsifier is no repeatable
forward-fold gain, particularly in the prespecified sparse-Tour subgroup, or
overall deterioration relative to matched Tour-only rating controls. Subgroups
are diagnostics and cannot replace the fixed full evaluation cohort.

Coverage check using only through-2022 history: among 5,056 player appearances
in eligible 2022 Tour matches, 648 have fewer than ten Tour matches in the
365-day interval ending at T−21. Of those, 560 (86.4%) have at least ten
Challenger/qualifying matches in that interval. This is coverage evidence, not
predictive evidence.

## Source and identity audit

The already-published inventory supplies 142,709 eligible 2006–2023 records:
44,164 Tour, 74,576 Challenger and 23,969 Tour/major qualifying matches. The
implementation uses the existing reconciled ATP identities and outcomes; it
does not scrape or alter the inventory. Of this inventory, the actual feature
run admits 73,861 Challenger and 23,903 qualifying matches, **97,764 additional
history records** in total, through its latest T−21 cutoff of 2023-10-16.
The 98,545 Challenger/qualifying inventory count includes later records that
cannot enter these target features. The audit found:

- Zero duplicate match IDs, missing/equal player IDs, nonbinary labels or
  nonpositive ranks in the development history.
- Each source/event has exactly one date and surface; all A/B IDs follow the
  published lexical orientation. 1,133 IDs appear in Tour and another source.
- 770 records have a tournament date in the late December immediately before
  their season label. These legitimate opening weeks retain their actual dates.
- Some qualifying dates represent the main-draw week. T−21 is conservative
  relative to the earlier qualifying date, but it cannot reconstruct exact
  match chronology. Qualifying/main-draw observations sharing a date are one
  simultaneous batch, never sequential same-event updates.
- The 2019 Challenger source has only 3,090 retained records, ending June 16.
  Missing results can distort rating and exposure trajectories; no later or
  final-test information fills this gap. Incomplete 2006 qualifying coverage,
  omitted unmatched IDs, carpet exclusion and positive-rank filters also
  persist. See [the original data audit](../data.md).
- Player-pair duplication is not sufficient to identify a duplicated match.
  In particular, the inventory contains two 2007 Buenos Aires RR B884/D330
  records. Raw inspection found different winners, scores, durations and point
  statistics; they are not identical duplicated rows. The repeated matchup
  has not been independently checked against an official draw in this task,
  and remains a source-verification limitation. This module does not silently
  rewrite the source population.

This family consumes results, not set-score strings; the earlier inventory's
four independently checked anomalous score strings therefore introduce no
score-parser dependency here. Filtering source and season happens before
historical outcome conversion. There is no 2024/2025 raw-data access.

## Fixed rating construction

All banks use ordinary Elo, `E_A = 1/(1+10**((R_B-R_A)/400))`, with fixed
`K=32` and equal-and-opposite increments `32*(y_A-E_A)`. Every game on a date
uses the ratings before that date; increments are summed deterministically
before one snapshot is stored. No tuning grid is introduced.

Three banks isolate historical source and initial-prior assumptions:

1. **Tour control:** only Tour matches, every newcomer starts at 1500.
2. **Common prior:** all three sources, every newcomer starts at 1500.
3. **Level-aware prior:** all three sources; first observed Tour / qualifying /
   Challenger appearance initializes at 1500 / 1400 / 1300. If first appearances
   share a date, the highest level observed on that date sets the prior. These
   numbers are fixed pragmatic assumptions, not fitted estimates or published
   cross-circuit conversion factors. The common bank controls their impact.

Crucially, the highest level in a player's **future career** is never used.
Before the first historical appearance, target queries use a neutral Tour
prior of 1500 in every bank. A player's first appearance at a new surface starts
that surface rating at their pre-date overall rating, providing hierarchical
initialization. Query-time surface strength is
`overall + n_surface/(n_surface+20)*(surface-overall)`. An unseen surface falls
back to overall exactly. This is explicit shrinkage, not a posterior obtained
by multiplying independent overall/surface evidence.

Ninety- and 364-day trends compare posterior ratings at T−21 and that cutoff
minus the specified days. Without an earlier posterior, trend is zero: entering
with a 1300 prior is not falsely called a deterioration from 1500.

## Schema and reproduction

`build_multilevel_features(targets, inventory)` returns **37 finite numeric
columns**, preserving the exact target index and order. Targets must be from
2009–2023; other seasons raise. The builder stores no labels in features.

- Three banks × overall, surface and blended rating differences: 9.
- Two joint banks × overall/blend differences from Tour, two trends and
  overall rating mean: 10.
- Three sources × 364-day recent log-count difference/sum and share difference:
  9. This window is T−364 through T−21 inclusive. Shares use five fixed
  pseudo-appearances in their denominator.
- Total/lower-level career log-count difference/sum, minimum total count,
  surface shrinkage-weight difference/sum, recent lower-level share difference
  and unseen-player count: 9.

Every `_diff` column is odd under A/B exchange. All other columns are even;
the corresponding names and construction settings are supplied in DataFrame
attributes. Exact Tour-only controls use this module's hierarchical surface
initialization, so they are distinguished from previous feature banks.

Run `.venv/bin/python -m unittest tests.test_multilevel_features -v`.
Eight tests cover the Elo hand calculation, surface shrinkage, same-date
initialization/update ordering, first-level future poisoning, inclusive cutoff,
sealed labels, target-label independence, future roster isolation, A/B and
historical-orientation symmetry, propagation through shared opponents, absent
surface fallback and invalid/held-out input rejection.

Full-cohort benchmark: **36,421 × 37 finite values in 2.774 seconds**, including
CSV input. The last target cutoff is 2023-10-16, so this actual feature build
uses 43,998 Tour, 73,861 Challenger and 23,903 qualifying historical matches;
later 2023 results are not required. No models were fit in this benchmark.

## Interpretation and remaining risks

Disconnected or weakly connected player populations make prior scales matter.
Joint Elo may overvalue repeated low-level wins, and ATP rankings already
summarize some lower-level results. Counts represent recorded coverage rather
than complete career exposure. The first observed source need not be a player's
first professional match. Same-date batching sacrifices true within-event
chronology to avoid invented timestamps. These are reasons to retain the
common-prior and Tour-only controls, not reasons to claim the new family works.

Selection belongs to the parent's declared forward training protocol. Reused
2022 and 2023 remain exploratory after previous rounds, even when the new
feature definitions were fixed before their new scores. Final-test evaluation
requires the separately frozen handoff; this module intentionally refuses it.
