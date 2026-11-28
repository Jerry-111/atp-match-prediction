# Freshness feasibility and the Round 8 timing audit

2026-11-27. This is a **coordinator-prompted, post-result investigation**, not
an independently generated Round 9 idea or a new approved experiment. The
scientific-brainstorming workflow separates the proposed mechanism, source
evidence and decision. Earlier rounds and their 2023 results were already
visible. No new model was fitted or selected here; no target, label, threshold,
Round 9 input or frozen Round 8 numerical result was changed.

## Question, mechanism and rival explanation

**Idea:** more recent, genuinely completed historical matches might describe
current strength better than the original T−21 history. This would be a
separate later-information condition, evaluated on all original 2,588 targets
with exact existing-model fallback where availability is unverified.

**Assumption requiring proof:** the historical result and any technical counts
were already available before the target prediction time. An earlier
tournament-start date alone does not prove this. A result improvement could
instead be leakage from overlapping events, retrospectively corrected dates,
selective timestamp coverage or different training data. Those alternatives
must be excluded before any predictive experiment. Nothing here predicts 70%.

## What the current sources establish

TennisMyLife documents `tourney_date` as the **week of the tournament**, not a
match start, finish or results-publication timestamp. Its historical records
are updated and corrected; `minutes` is match duration, not elapsed wall-clock
time including every interruption. There is no per-row historical publication
log in the pinned annual files. [Provider documentation](https://stats.tennismylife.org/tennis-match-database).

The published eligible 2006–2023 inventory contains 44,164 completed, ranked
Tour rows from 1,117 editions: 19,495 ATP250, 6,620 ATP500, 8,621 Grand Slam and
9,428 Masters matches. It excludes Davis Cup, Olympics, team events and Tour
Finals. Challenger history has 74,576 rows; Tour qualifying has 23,969. These
counts describe already-filtered inventory, not all professional matches.
Round 6's separately allowlisted extended pool also includes Davis Cup,
Olympics, team events and Finals; it must not inherit a one-week-event rule.
The known 2019 Washington round-dependent date correction already demonstrates
why a source date needs edition-specific validation.

Valuebetennis documents `date` as match date/time in UTC and the prices as
opening or final prematch quotes. It does **not** supply independent
`actual_start`, `completed_at` or quote-capture timestamps. Its settled export
is not a historical as-of snapshot. [Column documentation](https://www.valuebetennis.com/en/donnees.htm),
[collection methodology](https://www.valuebetennis.com/en/methodologie.htm).
The fixed outcome-blind links provide a time for 2,381/2,388 targets in 2021,
2,495/2,528 in 2022 and 2,502/2,588 in 2023. Thus even time linkage leaves 86
2023 matches without a timestamp; usable opening prices have different coverage.

**Counterevidence in the existing 7,341 Round 8 opening edges:** there are eight
same-player/event groups with distinct knockout rounds at exactly the same
UTC (16 appearances), and 14 successive own-round transitions whose reported
UTC does not increase. These counts overlap and must not be added. For example,
McDonald's Toronto R32 against Rublev (`2023-421:32`, source 623707) and R16
against Raonic (`2023-421:44`, source 624089) both carry 2023-08-10 18:00 UTC.
The ATP explicitly dates the Rublev match to Wednesday 9 August and the Raonic
match to Thursday 10 August. This is direct evidence that the field cannot be
universally interpreted as actual start time. [ATP Wednesday report](https://www.atptour.com/en/news/fritz-humbert-mcdonald-rublev-toronto-2023-wednesday),
[ATP Thursday report](https://www.atptour.com/en/news/ruud-davidovich-fokina-toronto-2023-thursday).

Negative source-UTC offsets from the nominal tournament date are not all errors:
Sunday starts and timezone differences are possible. The ATP's 2023 calendar
lists Adelaide starting 1 January, whereas its repository edition is dated
2 January. We therefore record offset discrepancies without automatically
rewriting individual dates. [ATP 2023 calendar](https://www.atptour.com/-/media/files/media-guide/2023/2023-atp-media-guide-intro-administration-tournaments.pdf).

## Actual impact on frozen Round 8

The read-only [timing audit](../../scripts/audit_market_timing_round8.py) joins
all 7,341 frozen graph edges to their original corrected target metadata. No
odds or outcomes are read. Source UTC minus the linked tournament date ranges
from −0.895833 days (−21h30m) to +14.083333 days (+14d2h); 399 rows have negative
offsets (110 in 2021, 170 in 2022, 119 in 2023), all less than one day.

For every existing target cutoff C=T−21, the audit checks whether any edge
satisfies `source_UTC <= C` but `linked_tourney_date > C`.

| Population | Targets | Distinct cutoffs | Snapshot-edge uses | Target-edge uses | Violating uses |
| --- | ---: | ---: | ---: | ---: | ---: |
| All development | 36,421 | 543 | 374,716 | 26,379,109 | 0 |
| 2021 H2 selection | 911 | 15 | 26,503 | 1,588,984 | 0 |
| 2022 gate | 2,528 | 36 | 125,650 | 8,676,837 | 0 |
| 2023 exploration | 2,588 | 35 | 209,701 | 15,307,076 | 0 |

There are 107 nonempty development cutoffs; 7,097 of the 7,341 source edges are
ever admitted by the latest target cutoff. Snapshot-edge counts count each
logical cutoff once, before repetition across six graph/surface settings;
target-edge counts weight each cutoff by its number of targets. The minimum
age of both the nominal tournament date and source timestamp for an admitted
edge is 21 days. The locked-pre2023 diagnostic is also checked separately.

**Located evidence:** the original nominal tournament-date guard is satisfied
for every existing Round 8 edge use. Adding both date conditions would remove
zero edges and therefore change no graph feature or probability. This does
not prove independently verified completion or quote availability at T−21.
The three-week buffer was not bypassed relative to the original repository
date policy; a universal real-time availability guarantee is not established.

The original registry, model bundle, features, OOF predictions, validation
probabilities, scores and selection remain byte-for-byte frozen. The protocol
and report now say **provider-reported UTC**, not verified actual UTC or a
proven upper bound. Exact reversible text replacements and old/new hashes are
recorded in [text_revision.json](../../results/round8/market_graph/text_revision.json).
The independent graph audit is rerun without fitting a classifier. The frozen
feature module's legacy “actual-UTC” wording is superseded by this clarification;
its algorithm and source hash are deliberately unchanged. Round 9's pure-history
inputs do not use the Round 8 graph and are unaffected.

Reproduce the metadata audit and numerical audit with:

```bash
.venv/bin/python -m scripts.audit_market_timing_round8
.venv/bin/python -m scripts.audit_market_graph_round8
```

## Why event-start shortcuts are insufficient

- A multiweek event may still be running after a later event begins. Even its
  planned end date is not a completion certificate: rain moved the 2011 US
  Open men's final to Monday 12 September. [USTA account](https://www.usopen.org/en_US/news/articles/2020-04-24/2020-04-24_us_open_classics_novak_djokovic_vs_rafael_nadal_2011_us_open_mens_singles_final.html).
- A competition label can span separate phases. The 2023 Davis Cup Finals had
  group play 12–17 September and knockouts 21–26 November. The current raw
  Davis Cup tie IDs must be validated separately; this example does not claim
  those two phases share one repository ID. [ITF preview notes](https://www.itftennis.com/media/10930/2023-davis-cup-finals-group-stage-preview-notes.pdf).
- A timestamp plus playing minutes does not establish wall-clock completion.
  Toronto's 10 August 2023 matches included a 94-minute rain interruption.
  [ATP interruption report](https://www.atptour.com/en/news/toronto-rain-thursday-2023).
- An appearance in a differently named event is not itself proof of completion
  of every match associated with the earlier source event. Team phases,
  qualifying/main-draw boundaries, rescheduling and identity/date errors need
  explicit treatment. Same-event strict own knockout paths already underlie
  Round 7; simply reusing them would not constitute a new information source.

## Minimal valid transition: proposed, not approved for fitting

Prefer an **edition-specific completion ledger** over a global shorter lag.
For an initial narrow allowlist of ordinary Tour knockout events, locate an
official completed-final report or final draw with a defensible publication
or completion-day upper bound. Separately verify the target event's earliest
possible main-draw start from its official calendar/order of play. If only a
local day is available, use its end in a verified timezone, or a conservative
whole-day bound covering the relevant timezone uncertainty; do not invent a
precise finish time. Accept the historical edition only when its certified
availability upper bound is strictly earlier than the target's conservative
start lower bound. Unknowns retain the original R4/full-cohort fallback.

This could add last-week results without needing current prices or complete
per-match timestamps. It would forgo some information and require manual or
auditable source collection. Keep team, multi-phase, Olympic, unverified
qualifying and delayed events out of this first allowlist until their specific
intervals are proven. A match-level alternative needs a verified later-round
completion witness or explicit finished-status archive; provider UTC alone
is insufficient, and final-round matches still need another completion source.

**Decision:** persist the timing correction and keep existing scores unchanged.
Defer fitting a freshness branch. A future go/no-go check should first report
completion-ledger coverage, counterexamples, fallback counts and timestamps
using only metadata, before selecting a finite learner menu. Select/tune only
on the pre-2023 development period, retain all 2,588 exploratory 2023 targets,
and keep 2024–2025 raw data sealed. Better apparent accuracy without certified
availability would falsify the validity of the experiment, not support the idea.

The bounded primary-source search checked provider column contracts, official
2023 calendars and dated match reports, the 2011 weather counterexample and
ITF phase dates. It did not establish complete availability coverage or a
research gap. No human domain-expert review or independent future test is
claimed; the coordinator's freshness prompt is an acknowledged anchor.
