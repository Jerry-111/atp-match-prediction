# Independent completion-evidence review and full source worklist

2026-11-28. This records an independent read-only review of the
[six-case pilot](round10_completion_feasibility.md). It does not authorize a
predictive experiment, change a history cutoff, or establish that the broader
worklist is available before play. The review and source lookup preceded this
artifact-writing task; reproduction uses no network access.

The [independent result](../../results/round10/completion/independent_review.json)
matches all six cases' source, target and newly covered match IDs, dates, counts
and direct-participant intersections. Arithmetic was implemented separately from
the original audit: the conservative clearance is `24 × (T − D in days) − 50`
hours. The target metadata SHA256 also matches. Only the six ID/date/player-ID
columns are parsed; full-file hashing does not expose its outcome columns to the
analysis. No final-year raw data was read.

| Transition | Independent clearance (hours) | Original T−21 already includes source | Newly covered target IDs |
| --- | ---: | --- | ---: |
| Rome 2021 → Roland-Garros 2021 | 286 | Yes | 0 |
| Halle 2021 → Wimbledon 2021 | 142 | No | 60 |
| Wimbledon 2021 → Bastad 2021 | −26 | No | 0 |
| Rome 2022 → Roland-Garros 2022 | 118 | No | 99 |
| Halle 2022 → Wimbledon 2022 | 142 | No | 45 |
| Wimbledon 2022 → Bastad 2022 | −26 | No | 0 |

Thus **four of six** date bounds pass; **three** source editions are newly
eligible under the stated publication assumptions, directly involving **204 of
544 distinct target IDs**. These are match counts, not unique-player counts or
an estimate of whole-season coverage. The source pool is the original eligible
completed/ranked Tour population, not every match in an official draw. The two
negative adjacent-day cases remain: overlapping conservative bounds mean the
evidence is insufficient under this policy, not that results were actually
unavailable. Rome 2021 remains a zero-increment control because the original
nominal-date cutoff already included it; official Sunday openings do not rewrite
that frozen comparator.

## What the independent source review establishes

Five dated completed-final reports were reopened: the participant's official
[Rome 2021](https://novakdjokovic.com/en/news/tennis/novak-with-runner-up-finish-in-rome/),
[Wimbledon 2021](https://novakdjokovic.com/en/news/tennis/novak-wins-historic-title-at-wimbledon-claims-20th-grand-slam-trophy/),
[Rome 2022](https://novakdjokovic.com/en/news/tennis/nole-overpowers-tsitsipas-to-win-sixth-rome-title/),
[Wimbledon 2022](https://novakdjokovic.com/en/news/tennis/impressive-novak-wins-fourth-consecutive-wimbledon-title/),
and the French federation's [Halle 2021 report](https://www.fft.fr/actualites/humbert-un-bleu-gagnant-sur-le-vert).
Their displayed historical dates and descriptions of completed finals agree
with the ledger. Current dated pages are not immutable historic publication
snapshots; none was obtained by this review.

The [Halle 2022 Tennis TV video](https://www.youtube.com/watch?v=jULQZp74uQY)
could not be fetched in the independent review, and an exact video-ID search
returned no results. Consequently its 19 June publication date remains the
original reviewer's indexed-primary observation, **not an independently
reverified timestamp**. The completed draw corroborates the finalists and
champion; by itself it establishes neither completion day nor first-publication
time. This limitation does not silently delete the case or promote its evidence.

The official Roland-Garros notices, both Bastad edition pages, Wimbledon date
materials and schedules were reopened. They support the target-opening dates
recorded in the pilot. The current Wimbledon Day 1 schedules contain completed
results; the [2022 PDF](https://www.wimbledon.com/en_GB/scores/2022/schedule/pdf/schedulePDF8.pdf)
is marked as updated on 5 July. Such files corroborate the historical opening
date but are not evidence that this precise file was available before Day 1.
No date was inferred from a target winner. The blocked ATP yearbook was not
requested in this independent review or artifact-writing task.

## Complete nominal-date sampling frame, not qualified availability

Before any further source adjudication, define candidate pairs by
`source_nominal_date > target_nominal_date − 21 days` and
`source_nominal_date <= target_nominal_date`, excluding the target edition itself.
This deliberately retains same-week and overlapping events. A nominal date is
not a certified actual first-match date. Official date corrections and completion
or publication evidence may later reject pairs; this worklist admits none for
prediction. It covers the existing eligible Tour metadata, not all tennis events.

| Target season | All target editions / matches | Candidate pairs | Distinct candidate source editions | Target editions with any pair |
| --- | ---: | ---: | ---: | ---: |
| 2021 | 60 / 2,388 | 214 | 57 | 59 |
| 2022 | 67 / 2,528 | 290 | 65 | 67 |
| 2023, unqualified extension | 64 / 2,588 | 259 | 62 | 64 |

The 2021–2022 frame contains **127 target editions and 504 pairs**.
[source_frame.csv](../../results/round10/completion/source_frame.csv) preserves
every pair, both event IDs, nominal dates, cutoff and source/target match-ID lists.
[target_editions.csv](../../results/round10/completion/target_editions.csv) also
retains **2021-499**, the one target edition with zero candidate sources in this
window. Hence the pair file touches 126 target editions while the complete
target sampling frame retains all 127. No target disappears because it has no
candidate evidence.

Only the original **six** purposive cases received a completion-source review.
Five occur in this new-information pair file; Rome 2021 → Roland-Garros 2021 is
outside it because its nominal dates, 8 May and 31 May, are **23 days apart**
and already satisfy T−21.
The `prior_six_case_review` and `prior_case_date_bound_status` fields preserve this
distinction. Every row still has `admitted_for_prediction=False`.

The same-rule 2023 extension is separate:
[source_frame_2023_unqualified.csv](../../results/round10/completion/source_frame_2023_unqualified.csv)
and [target_editions_2023_unqualified.csv](../../results/round10/completion/target_editions_2023_unqualified.csv).
Its **259 pairs have zero completed source reviews**. No coverage or predictive
gain is claimed for this extension; it is a future worklist. All four CSVs
contain metadata and identifiers, without labels, scores, statistics or market
prices. Frame construction does not use player overlap or model errors to select
which events deserve investigation.

## Smallest next source task

1. Freeze the full candidate source-edition list and target-opening evidence
   standard. Obtain one source certificate per edition and reuse it across all
   candidate targets; preserve unavailable, ambiguous and negative cases.
2. Record completion and publication separately, including URLs, content hashes,
   retrieval times, displayed historical dates and any independently archived
   evidence. For established bounds, require
   `max(completion_upper, publication_upper) < target_start_lower`. If publication
   is only inferred from current dated reports, keep that assumption explicit;
   do not call the record an exact as-of snapshot.
3. Initially use only existing result identities/outcomes. The pilot does not
   certify initial publication or subsequent correction of technical statistics.
4. Audit the complete coverage first, then freeze any matched stale/fresh
   experiment. Keep the same rating algorithm and deterministic event-batch
   update rule, original targets and exact T−21 fallback for missing evidence.
   Do not choose source additions by examining individual 2023 mistakes.

No empirical fitting is authorized by this document. The ongoing 70% goal,
unchanged full 2,588-target cohort, threshold 0.5 and sealed 2024–2025 remain.

Reproduce all metadata checks and worklists offline:

```bash
.venv/bin/python -m scripts.audit_completion_source_frame_round10
```

The [independent script](../../scripts/audit_completion_source_frame_round10.py)
imports no function from the original completion audit. It verifies that the
original ledger's code hash still matches, stores separate independently
calculated ID lists and bounds, and binds its source, this review document, the
original ledger and all four output CSVs by SHA256. Reproduction validates local
arithmetic and identifiers; it cannot replay the earlier human reading of web
pages or recover their historical publication state.
