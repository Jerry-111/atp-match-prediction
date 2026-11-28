# Round 11: independent review of completed-event availability

2026-11-28. **Design and metadata review only; no model fit, prediction-error
inspection, source admission or change to the frozen T−21 artifacts.** The
fixed 2,588 targets, threshold 0.5 and sealed 2024–2025 remain. This follows the
scientific-brainstorming workflow: two initial mechanisms were sent to the
parent independently, before reading the new source probe or discussing its
proposal. Literature/source checking and adversarial refinement followed.
The criteria are identifiable information gain, credible time semantics,
whole-cohort feasibility and useful negative results, not agreement among agents.

**Recommendation:** permit a separately labelled, retrospective availability
experiment using full primary completion evidence and explicit publication
assumptions. Do not require an immutable archive for every exploratory source,
but do not describe this tier as a strict as-of reconstruction. Use one common
admission rule, retain all missing/negative cases and compare identical stale
and fresh algorithms. More accurate time bounds may rescue useful recent
history; venue timezone knowledge alone does not establish publication time.

## What a certificate must mean

For a source edition, distinguish actual completion `C`, the time the admitted
result content became publicly available `P`, and the target edition's first
competitive main-draw point `T`. The target boundary applies to **every** target
row of that edition, including later rounds. It is not each player's first
match, qualifying's first day, a nominal Monday, or the latest scheduled session.

Record a completion bound, a publication bound or explicitly assumed bound,
and a lower bound `L <= T`. Admit only when both source bounds precede `L`.
A dated completed-final report can establish that the final had ended by its
publication, under its date/content assumptions. It does not by itself establish
when every earlier match's exact archive row or technical statistic was first
published. The first experiment should therefore add only the existing eligible
match identities and winner outcomes, under a disclosed assumption that public
completed-draw outcomes were available and stable by the certificate time.
Do not add technical statistics, updated rankings or current-edition results.

| Evidence tier | What it supports | Permitted interpretation |
| --- | --- | --- |
| A: archived content before the target | A pre-target capture contains the complete admitted result content and completed status; its timestamp and coverage are checked | Strict as-of for those captured fields and that source, subject to the archive's own provenance; a final-only capture does not certify the full draw |
| B-time: primary completed-final text plus an identifiable public timestamp | A readable official report/post identifies the edition and completed final; the timestamp is either explicit with timezone or a content-linked post ID | Exploration under stated timestamp, immediate-publicity and content-version assumptions; exact clock precision does not upgrade this to tier A |
| B-day: primary completed-final text with a defensible completed-by civil day | A report explicitly identifies a local completion day, or its publication day has a supported timezone; target opening day is independently supported | Exploration under public-result availability/stability assumptions; use the appropriate local or conservative global interval and retain unresolved publication semantics |
| C: calendar range, current draw/replay listing, search snippet alone, ambiguous post, denied page | Edition identity or a discovery lead, but insufficient completion/publication evidence | Keep in the worklist; no admission from this evidence alone |

For B-day, distinguish two claims in the ledger. A source with a supported
publication-zone/day has a **conditional publication bound** from that day.
A source with a supported event-local completion day but unknown publication
zone can only use an **assumption that the eligible public result outcomes were
available by the end of that completion day**. This is a reasonable bounded
research assumption for official completed knockout results, not an observed
publication timestamp. A protocol must choose and name this assumption before
scoring; unknown timezone must never silently become the venue timezone.
Where the completion day itself is ambiguous, this assumption cannot repair it.

Required fields include edition identity, evidence URL and readable assertion,
retrieval time/content hash when actually captured, displayed dates, venue and
publication zones separately, bound type/inclusivity, any archived capture,
content/version scope, evidence tier, assumptions, and acceptance/rejection
reason. A summary hash is not a downloaded source-body hash. One edition's
certificate may be reused across target editions; a blocked URL stays blocked.
Conflicting dates, uncertain edition identity, a postponed/unfinished final,
overlapping bounds or an unsupported target opening require rejection/fallback.

## Local time helps, with two important counterexamples

The [IANA database](https://www.iana.org/time-zones) models historical UTC offsets
and daylight-saving changes. Its [technical description](https://www.iana.org/time-zones/tz-link)
supports location-specific historical conversion, not inference of a website's
publication timezone. Pin the tzdb version or zone-file hashes and verify the
**edition's actual venue**, including temporary relocations; a persistent
tournament suffix or today's venue is insufficient. Construct each local
midnight and convert it to UTC. Do not replace a civil day with 24 hours across
a daylight-saving transition. Resolve ambiguous boundaries conservatively, or
reject them.

If a source is certified completed during local date `D`, its completion is
strictly before the next local midnight `U`. If the target first plays on local
date `S`, its first point is at or after that day's midnight `L`. Thus
`C < U <= L <= T` is safe: **equality is acceptable for an exclusive upper
bound**. An inclusive bound `C <= U` still requires `U < L`. Apply this to both
completion and publication; if either maximal bound is inclusive, equality
does not establish strict prior availability. Store the flags rather than
globally changing the old comparison from `<` to `<=`.

The old global date intervals deliberately had approximately 50 hours of
separation overhead. Local zones remove much of that loss, but **same continent
is not same timezone**. Independently calculated examples using `zoneinfo`:

| Assumed local dates | Source exclusive end bound, UTC | Target inclusive start bound, UTC | Result |
| --- | --- | --- | --- |
| Wimbledon 11 July 2021, London → Bastad 12 July, Stockholm | 11 July 23:00 | 11 July 22:00 | Still overlaps by 1 hour |
| Wimbledon 10 July 2022, London → Bastad 11 July, Stockholm | 10 July 23:00 | 10 July 22:00 | Still overlaps by 1 hour |
| Same-zone Sunday completion → Monday opening | Same UTC instant | Same UTC instant | Passes only with the supported exclusive completion/publication bounds |

The first two transitions therefore need either a stronger source timestamp or
a supported target Day 1 not-before hour. Do not invent a universal daytime
start. A clearly identified first-day order of play can provide that lower
bound under an explicitly retrospective schedule-consistency assumption; an
updated schedule is not an immutable pre-event publication record. Delays do
not violate a not-before lower bound, but earlier rescheduling or a missing
earlier court/session can. A mere notice that some main-draw matches occur on
Monday does not establish that none occurred on Sunday.

Australian Open 2022 is the necessary cross-midnight control. The official
[champion report](https://ausopen.com/articles/match-report/alone-top-nadal-wins-record-21st-major-singles-title)
currently displays 30 January and describes the Sunday-night final, while the
[official analysis](https://ausopen.com/articles/features/analysis-how-nadal-won-final)
displays 31 January. An on-site
[ABC report](https://www.abc.net.au/news/2022-01-31/australian-open-rafael-nadal-beats-daniil-medvedev-in-final/100791418)
explicitly places the finish after 1 a.m. AEDT. The latter is a contemporaneous
eyewitness broadcaster report, not an ATP timing feed. Scheduled final date or
“Sunday night” must not produce a false Sunday-local-midnight completion bound.
Under day-only evidence for 31 January, the next Melbourne midnight is
31 January 13:00 UTC; this can be too late to qualify an early European Monday
target despite the actual final having ended much earlier. That is a precision
limitation, not permission to move the completion date back.

## Snowflake: useful timestamp evidence, not an automatic upper bound

[X's ID documentation](https://docs.x.com/fundamentals/x-ids) says IDs contain a
creation timestamp. Twitter's own
[archived Snowflake implementation](https://raw.githubusercontent.com/twitter-archive/snowflake/snowflake-2010/src/main/scala/com/twitter/service/snowflake/IdWorker.scala)
uses epoch `1288834974657` milliseconds and a 22-bit lower payload. For a string
status ID, decode with exact integer arithmetic:

```python
created_ms = (int(status_id) >> 22) + 1288834974657
```

The AO report's currently readable embedded post links to status
`1487793710796623872`. This decodes to `2022-01-30T14:23:57.181Z`, or
`2022-01-31T01:23:57.181+11:00` in Melbourne. The visible short caption alone is
too generic to certify a completed final. This is a timestamp/boundary example,
**not a newly admitted certificate**. No X login or private API was used; the
link and caption were read from the public official report.

Object creation does not alone bound public availability from above: visibility
could be delayed/private, the link could identify a different version, and a
current embed could contain changed text. A full primary report containing an
explicit completed-final post from an identified official account can qualify
as **B-time**, provided the protocol assumes the visible result content belongs
to that ID/version and was publicly visible at creation. Preserve those
assumptions as fields, not as a “verified publication UTC” claim. Without the
content/account/edition connection, the ID is only auxiliary evidence.

The [September 2022 launch notice](https://blog.x.com/en/product/twitter-new-edit-tweet-feature-only-test)
describes the introduction of editing, and current
[developer documentation](https://docs.x.com/x-api/fundamentals/edit-posts)
states that edits create new IDs. Publicly visible version evidence can improve
the link, but cannot be presumed. Do not add a universal “30-minute” or “1-hour”
buffer to fix unknown content history: current developer and
[help documentation](https://help.x.com/en/using-x/edit-post) even describe
different edit windows. Neither current policy establishes a historic
content/publication guarantee. A timestamp-only comparison also needs strict
separation from the target; do not use millisecond rounding to admit ties.

## Is the nominal-date source frame complete enough?

Metadata-only reproduction reads `match_id`, `season`, `tourney_id` and
`tourney_date` from the fixed 36,421-row development table. Its SHA256 remains
`9bcb3ad4b0fbe745b23b776a750256f92ab9acc9fd1cb57aed0a16cef86435d2`.
No outcomes, prediction columns or final-year raw files are parsed.

| Target year | Targets / editions | Candidate pairs | Same nominal-date pairs |
| --- | ---: | ---: | ---: |
| 2021 | 2,388 / 60 | 214 | 52 |
| 2022 | 2,528 / 67 | 290 | 80 |
| 2023, still unqualified | 2,588 / 64 | 259 | 78 |

The [frame builder](../../scripts/audit_completion_source_frame_round10.py)
constructs events across all available seasons before comparing dates. It does
**not** restrict a source to the target's season. There are no cross-season
pairs in these particular 21-day windows, and no season/calendar-year mismatch
among the 2021–2023 target edition dates. This observed zero is not a reason to
introduce an equality-of-season filter. The zero-source 2021-499 target remains
in the separate edition table.

Three boundary checks matter before claiming complete fresh-history coverage:

1. **Sunday or other non-Monday opening.** A Monday nominal target can actually
   begin Sunday. Keep its frozen stale cutoff, but apply the verified earlier
   target boundary to every fresh source. Do not infer actual opening from
   nominal weekday. Conversely, a target can begin after its nominal date; the
   new [2021 probe](round11_2021_source_probe.md) records this distinction for
   Delray Beach. Sources with nominal dates later than the target nominal date
   can be omitted by the current pair rule if those nominal dates do not order
   actual completion/start. This is a possible frame omission, not evidence
   that any such source actually qualifies.
2. **Multiweek sources.** An edition with nominal start at or before T−21 is
   already in the stale information set, even if it completed recently. It is
   not a missing *incremental* source: Rome 2021 is the worked zero-increment
   example. Audit unusually long/postponed editions for any stale-baseline
   completion-after-target conflict; do not silently remove them and attribute
   the resulting change to freshness. T−21 means 21 days after nominal start,
   not 21 days after completion.
3. **Cross-year and source population.** Compare real dates across seasons and
   keep prior December editions where applicable. The source population is
   only the original completed/ranked Tour population; editions with no eligible
   rows, other competition levels or omitted matches are outside this
   experiment. Extending that population would be a different intervention.

A small completion check should first establish target openings and compare
the full eligible edition catalog around each boundary, not blindly expand the
admission lag. Any additional candidate pairs must be recorded and frozen
before model scores, with the reason the original nominal frame missed them.
The current same-week candidates are deliberately overinclusive. Frame
membership alone never qualifies a source. Keep the existing R10 CSVs immutable
and save any expanded worklist separately.

## At most two paired algorithms

These are proposals for protocol approval after the source pass, not permission
to fit now. Both preserve the historical source filter, target cohort, A/B
orientation, threshold and a deterministic shared update rule. A clean null
result is useful: it would show that this qualified freshness increment does
not improve that model, rather than suggesting another broad parameter grid.

**A11-1: fixed event-batched Elo replay.** Use the existing K=32, initial 1500,
400-point base-10 logistic rule. For each target edition, the stale state contains
exactly the eligible original history with nominal date at or before its frozen
T−21 cutoff. Fresh starts from that same state and admits only disjoint,
certificate-qualified source records after the stale cutoff. Process additional
nominal dates in order; all records sharing a nominal date use pre-batch ratings,
and sum their updates before advancing. This retains the established no-invented-
within-event-order convention. Report raw overall Elo stale/fresh as the one
paired estimator, with the existing R4 forecast as an unchanged contextual
benchmark. There is no new optimizer or K search.

Do not carry a mutable fresh state between target queries: a record once added
fresh will later enter a stale prefix and could be counted twice. Query-specific
prefix-plus-unique-extra-ID replay avoids that. No-extra queries return the
saved stale value exactly. Test permutation within a batch, identity uniqueness,
future/current-edition poisoning and A/B complementarity. Because rating updates
can propagate through opponents, direct participant intersections are only a
coverage description, not the complete set of affected targets. Improvement
over raw stale Elo would not by itself establish improvement over R4.

**A11-2: one shared rank/rating learner, paired input intervention.** Fix the
odd inputs to log-rank difference, overall K=32 Elo difference and the existing
20-match-shrunk surface K=32 Elo difference. Use one symmetric ridge-logistic
recipe, fixed C=.01, no intercept, train-only zero-centred RMS scale and the same pre-year,
21-day-purged training rows. Fit **one** stale-history model per outer year,
then apply its identical coefficients and scaling to stale and fresh versions
of those three fields. Fresh recomputes the surface count/shrinkage consistently;
ranks remain unchanged. Missing evidence returns the same stale vector and
prediction exactly. This tests incremental information while retaining rank
and surface strength, with a disclosed assumption that the learned mapping
transports to fresher covariates.

The original independent A11-2 suggestion allowed separately fitted stale and
fresh calibrators. Adversarial review narrowed it to a shared mapping: separate
fits conflate information change with coefficient/scale refitting. Updating a
few of the 326 old features inside a complex frozen model would likewise leave
correlated history fields at incompatible times; this compact design avoids that
unmeasured intervention. A separately trained fresh pipeline could be a later,
explicitly different estimand, not an undisclosed replacement if this pair fails.

If source qualification exists only for 2021–2022, do not fabricate a 2020 fresh
validation fold. Freeze the two structures/selection rule before any comparison,
use 2021 for the one early choice and 2022 for the practical gate, then freeze
the method before the qualified full 2023 run. If 2021 coverage is inadequate,
report a feasibility failure rather than choose from 2023. Report both paired
whole-cohort accuracy and log loss, source/target certificate coverage and
changed-prediction counts. Coverage-only subgroups are descriptive, never the
70% denominator. Unknown evidence keeps exact stale fallback, not dropped rows
or neutral probability. R4 stays a separate unchanged reference; stale fallback
in these pairs does **not** mean fallback to R4.

The reused 2022 gate is a practical development rule, and repeated 2023 use
remains exploratory. Neither improved evidence timestamps nor a paired design
restores an untouched test set. No source, timestamp precision, learner weight
or missingness rule should be chosen by individual 2023 mistakes.

## Bounded source-pass decision after this review

The parent subsequently chose the already chronological **first ten candidate
source editions per year**, rather than waiting for every edition to be
qualified. Unknown publication zones retain the conservative global date bound;
official post clocks remain auxiliary and do not trigger additional network
work. The local-time and post-clock refinements above are reviewed alternatives,
not newly adopted admission rules. The next limited experiment may use only
qualified source pairs together with independently established earliest target
dates. Unreviewed, unresolved and negative sources remain visible, and every
original target remains with exact stale fallback where no increment qualifies.

This is enough to test a bounded early-season freshness mechanism; it is not
a full-season refresh or a representative estimate of certificate availability.
First count the admitted **incremental** source IDs and affected target IDs
without reading prediction errors. A completed-source count is not an increment
count. For example, a typical two-week source finishing Sunday can fail the
global date bound for the following Monday, while the next Monday is already
21 days after its nominal start and therefore includes it in the stale baseline.
It may supply no newly eligible row despite excellent completion evidence.

Whole-year 2021 and 2022 paired comparisons remain the selection/gate denominator.
Report first-ten coverage and full-cohort uncertainty; do not change the source
policy or select additional editions because the bounded pass has low coverage
or disappointing accuracy. A zero/very small incremental population is a useful
feasibility result, not a reason to claim a refreshed model. This update fixes
the scope before any predictive fit and does not alter the R10 frame or ledger.

## Review limits and decision record

The later [frozen experiment protocol](round11_freshness_protocol.md) was reviewed
against these proposals: two algorithms, three shared logistic fits, no recency
weighting, four probability columns, 2021-only delta selection, a 2022 paired
gate and no pooled reselection. The existing symmetric wrapper fits its scaler
on both orientations of these three odd columns, giving the intended
zero-centred training scale. No protocol blocker was found.

A separate [2023 probe review](../../results/round11/completion/independent_2023_probe_review.json)
records the source checks and the reviewed input hashes. Five of the six reported
primary completion reports were independently reopened: Adelaide 1, Auckland,
Adelaide 2, Dallas and Rotterdam. Pune returned 403 and was not retried; it
retains the original reviewer's full-text observation rather than a claim of
independent re-verification. The other four unresolved first-ten source
editions remain unresolved.

Four of five target statements were independently reopened. AO explicitly
separates qualifying from the announced 16 January main draw; Marseille calls
20 February the first main-draw day; Rio lists 18–19 February qualifying and
20–23 February opening rounds. Dubai's official 25 February draw report says
all 16 opening R32 matches were drawn and the competition remained future.
That supports the deliberately early **draw-day lower bound**, not a claim that
main play began on 25 February. Acapulco's article-body retrieval timed out in
this review. The original reviewer observed the 31 January announcement of
27 February–4 March; its use retains the stated schedule-consistency assumption
and is not described as independently verified actual opening. These checks
evaluate source semantics; only the separate qualification ledger admits pairs.

This review directly inspected the existing frame code, recomputed the metadata
counts and timezone examples, read the primary IANA/X documentation, and read
the current official AO reports/embedded link. It did not verify every edition,
recover archived historical content, inspect any individual 2023 prediction,
fit a model or open a final-year raw file. The 2021 probe was read only after
the initial independent mechanisms were recorded. Its unresolved/denied sources
remain unresolved. A requested Kovalchik repository PDF returned access denied;
no workaround was attempted and no new paper claim relies on that response.

Proceed first with a frozen source-certificate policy and a bounded chronological
qualification pass. Tier B is scientifically usable with its assumptions visible;
Tier C is not repaired by optimism, geographic proximity or a bare status ID.
Keep all original artifacts and unsuccessful cases so the next teammate can
distinguish better evidence, a changed information regime and actual predictive
gain.
