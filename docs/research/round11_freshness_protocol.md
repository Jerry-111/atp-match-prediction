# Round 11: bounded completed-event freshness experiment

Protocol fixed before predictive fitting on 28 November 2026. The independent
[availability review](round11_availability_review.md) supplies the competing
mechanisms and their limitations. The source probes are a fixed chronological
first ten editions in each of 2021, 2022 and 2023, ordered by nominal start then
lexical ID. This is a limited early-season mechanism experiment, not a claim
that every recent result has been reconstructed. No individual prediction
errors determined that sample or the evidence decisions.

## Information intervention

The 2023 denominator remains the original 2,588 rows, classification threshold
0.5. 2021 and 2022 likewise retain all 2,388 and 2,528 rows. The 36,421-row
development table and existing completed/ranked Tour history population are
unchanged; 2024–2025 outcomes stay sealed. All 2023 measurements are exploratory.

The stale information set is the original eligible history with nominal date
at or before target nominal date minus 21 days. A fresh query adds only disjoint
eligible original match IDs in a qualified source edition after that cutoff.
No current-target-edition result, technical match statistic, new ranking,
Challenger match, unfinished match or extra target is introduced. Earlier
matches in the completed source draw are assumed public and stable with the
reported completed final. A final report is not row-by-row historical archive
certification. The unchanged stale comparator inherits its original timing
assumptions; this experiment does not rewrite that baseline.

Source qualification is **Tier B, retrospective under explicit assumptions**:
readable primary text identifies a completed final (or an explicitly later
completed-result upper day), with a defensible reported day. Let D be the later
of completed-by day and displayed publication day, resolving known conflicts
conservatively. Assume the eligible public outcomes were available, stable and
described by that dated report within the global civil-day interval. The
source completion/publication upper bound is D at 00:00 UTC plus 36 hours.
Unknown publication zones remain unknown; no venue-zone substitution is made.
This assumption does not make a current page an immutable historical snapshot.
Search snippets, schedules alone, denied text, generic social captions and
Snowflake creation times alone cannot qualify a source.

Each target requires an independent earliest-day lower bound S. An explicit
main-draw opening/Day 1 statement qualifies. A readable official event-period
statement, whether an advance notice or retrospective edition record, can
supply a lower day under the declared assumption that main play was not
advanced before that day; a draw announcement explicitly placing every opening
match in the future can supply its draw day as a deliberately earlier lower
bound. These are retrospective schedule-consistency assumptions, not claims
about first publication of the exact current page. The target lower bound is
S at 00:00 UTC minus 14 hours. Admit only source upper **strictly less than**
target lower. Do not adopt local-zone equality or social-post clocks in this
bounded run. Conflicts, overlap and unsupported dates preserve exact stale
fallback. No date policy is relaxed after coverage or scores are inspected.

The R10 candidate-pair frame is immutable. All pairs, rejected and unreviewed
sources, and all target editions remain recorded. This nominal frame may omit
rare date-misaligned candidates; no exhaustive fresh-history claim is made.
The separate metadata-only qualification script saves source/target IDs and
bounds, direct participant coverage and input hashes before any prediction.
Source-probe `admitted_for_prediction=false` remains an accurate discovery
status; only the separately frozen qualification ledger authorizes this run.

## Exactly two paired algorithms

1. **Overall Elo:** original K=32, initial 1500, base-10 scale 400. Read stale
   ratings at T−21; reconstruct fresh from that prefix plus unique admitted
   records. Apply existing nominal-date batches in order, all matches of a
   date using pre-batch ratings. No invented within-event sequence, K search,
   decay tuning or mutable fresh state shared across target queries.
2. **Shared rank/rating logistic:** original odd columns `log_rank_diff`,
   `elo_k32_diff`, `shrunk_surface_elo_k32_diff`; original 20-match surface
   shrinkage recomputed consistently for fresh inputs. Use SymmetricClassifier,
   C=.01, no intercept, train-only StandardScaler, mirrored half-weight rows,
   no recency weighting. One stale-history fit per prediction year, 2009 onward
   pre-year training with the original 21-day purge. Apply identical coefficients
   and scaler to both stale/fresh matrices. This explicitly assumes the mapping
   transports to fresher covariates; it does not test a separately retrained
   fresh pipeline.

There are exactly three new fits (2021, 2022, 2023) and four probability columns.
No mixtures, calibration search, threshold search, new complex model or update
to existing 326-field models is allowed. No-extra queries use the exact stale
vector and prediction. Existing R4 and R5 predictions are contextual references,
not the stale fallback of these two paired algorithms.

## Selection, evaluation and stopping this bounded experiment

On the **whole 2021 cohort**, an algorithm is eligible only if fresh accuracy
improves strictly and fresh log loss is no more than .005 above its own stale
control. Choose the largest accuracy gain; ties choose the lower loss delta,
then lexical algorithm name. If neither qualifies, freeze no freshness choice.
This is a mechanism-selection rule, not a new global champion selector.

For the single selected algorithm, the **whole 2022 cohort** gate requires
accuracy gain at least .005 (0.5 percentage points) and loss delta at most
.005 against that same stale algorithm. No selection means failed gate.
Freeze choice and gate before fitting/scoring 2023. Retain existing historical
recommendations if this gate fails. Passing this local gate still does not
establish superiority to R4/R5 or constitute SOTA; all contextual comparisons
must be reported. No pooled reselection and no switch to the best 2023 row.

Report all four estimates on full cohorts, correct counts, log loss, Brier,
paired changes, date-block paired intervals, number of eligible editions and
added unique source IDs, exact-fallback rows, changed probabilities and changed
classifications. Covered-only results are descriptive and cannot be the 70%
denominator. Publish coverage bounds on possible accuracy gains where useful.
The selection and intervals are not corrected for the extensive earlier search;
the reused 2022 gate is development evidence only.

After the four fixed estimates, stop this experiment even if negative. Save
three models, matrices/probabilities, protocol/source/code hashes, independent
reload and feature audit, and a reproducible report. The broader optimization
goal remains active unless actually achieved and delivered. A weak bounded
effect should motivate a distinct next hypothesis, not an unregistered sweep.
