# Round 10: six completed-event transitions, before predictive modeling

2026-11-28. This is a source and metadata feasibility result, **not a new model
score**. No history cutoff, probability artifact or recommendation changes.
The six source editions were specified by event/calendar class before any
prediction comparison: Rome, Halle and Wimbledon in 2021 and 2022, followed
respectively by Roland-Garros, Wimbledon and Bastad. The pairs include one-week
and two-week sources and two adjacent-day boundary cases. They are purposive
examples, not a random estimate of whole-season availability.

## What the date evidence establishes

Dated completed-final reports from a participant's official website establish
the Rome and Wimbledon observations. The French federation supplies Halle 2021.
Halle 2022 uses displayed publication metadata from the ATP's verified Tennis TV
final highlights, corroborated by the tournament's completed draw. The draw was
downloaded and visually checked: its rightmost box names Hurkacz as champion,
connected to Medvedev and Hurkacz as finalists. The scheduled 13–19 June range
alone would **not** establish an actual completion date.

| Source edition | Completion day | Primary completion evidence | Target earliest main-draw date and primary source |
| --- | --- | --- | --- |
| Rome 2021 | 16 May | [Participant's completed-final report](https://novakdjokovic.com/en/news/tennis/novak-with-runner-up-finish-in-rome/) | 30 May: [Roland-Garros revised main-draw dates](https://www.rolandgarros.com/en-us/article/new-dates-tournament-2021-postponed-by-one-week) |
| Halle 2021 | 20 June | [French federation final report](https://www.fft.fr/actualites/humbert-un-bleu-gagnant-sur-le-vert) | 28 June: [ITF Wimbledon edition](https://www.itftennis.com/en/tournament/wimbledon/gbr/2021/m-sl-gbr-01a-2021/draws-and-results/) and [official Day 1 schedule](https://www.wimbledon.com/en_GB/scores/2021/schedule/pdf/schedulePDF8.pdf) |
| Wimbledon 2021 | 11 July | [Participant's completed-final report](https://novakdjokovic.com/en/news/tennis/novak-wins-historic-title-at-wimbledon-claims-20th-grand-slam-trophy/) | 12 July: [ATP Tennis TV Bastad edition](https://www.tennistv.com/tournaments/316_2021/bastad) |
| Rome 2022 | 15 May | [Participant's completed-final report](https://novakdjokovic.com/en/news/tennis/nole-overpowers-tsitsipas-to-win-sixth-rome-title/) | 22 May: [Roland-Garros advance ticket notice](https://www.rolandgarros.com/en-us/article/roland-garros-2022-opening-official-ticketing) |
| Halle 2022 | 19 June | [ATP Tennis TV published final highlights](https://www.youtube.com/watch?v=jULQZp74uQY), [official final draw](https://www.terrawortmann-open.de/fileadmin/Gerry_Weber_Open/Ergebnisse/Rueckblick/Hauptfeld_Einzel.pdf) | 27 June: [official Wimbledon access guide](https://www.wimbledon.com/pdf/CH_2022_Easy_Access_Guide_v8_DIGI.pdf), [Day 1 schedule](https://www.wimbledon.com/en_GB/scores/2022/schedule/pdf/schedulePDF8.pdf) |
| Wimbledon 2022 | 10 July | [Participant's completed-final report](https://novakdjokovic.com/en/news/tennis/impressive-novak-wins-fourth-consecutive-wimbledon-title/) | 11 July: [ATP Tennis TV Bastad edition](https://www.tennistv.com/tournaments/316_2022/bastad) |

Evidence strength is deliberately explicit in the
[machine-readable ledger](../../results/round10/completion/source_feasibility.json).
Some target dates and the Halle publication date were inspected through indexed
primary-source metadata; they are not claimed to be complete audiovisual/PDF
inspections. The other dated reports and the 2022 Bastad page were opened as full
text. Current article headers, descriptions and navigation may have changed.
No independently archived historic publication snapshots were obtained.

The inference is conditional on these displayed historical dates being correct
and on public knockout outcomes being available by the completed final.
Completion evidence does **not** certify when every technical statistic was
first published or later corrected. A future pilot should initially use only
historical result identities/outcomes, and retain those assumptions. It must
not label these sources as proven exact machine-readable as-of snapshots.

The [ATP 2023 yearbook's 2022 final-results section](https://www.atptour.com/-/media/files/media-guide/2023/2023-atp-media-guide-rankings-stats-results.pdf)
was located in search and could potentially reduce manual source work. Its PDF
returned 403, so its layout/date semantics were not inspected or used in this
ledger. No proxy, mirror or repeated request bypass was attempted. Scheduled
calendars and the previously disputed provider UTC field remain insufficient.

## Conservative date-only bounds

Fix these rules before counts: source completion date `D` implies the conservative
upper bound `D at 00:00 UTC + 36 hours` (end of that day in UTC−12). Target earliest
main-draw date `T` implies the lower bound `T at 00:00 UTC − 14 hours` (start of that
day in UTC+14). Admit only when the upper bound is **strictly earlier**. These
loose global bounds intentionally avoid inferring actual match hours or venue
zones. They reject adjacent-day transitions; that is insufficient evidence under
this policy, not a finding that those outcomes were actually unavailable.

Use official Sunday main-draw openings for Roland-Garros even when the frozen
repository nominal date is Monday. Preserve the original nominal-date T−21
rule when determining what history the baseline already possessed. Do not
silently move the target cutoff or claim a fresh row that was already eligible.

## Reproducible metadata counts

The [audit script](../../scripts/audit_completion_round10.py) checks the fixed
development-table SHA256 and reads only ID, season, event, date and player-ID
columns. It does not load target outcomes, technical statistics or market data.
Only 2021–2022 editions enter this pilot; no final-year raw file is opened.
The original completed/ranked Tour target inventory is also the counted source
pool. “Either” and “both” mean direct participant intersections, not all indirect
rating propagation. Exact source and target IDs are saved in the ledger.

| Transition | Eligible source rows | Original target rows | Date-bound clearance / hours | Either / both direct participants | Newly covered targets |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rome 2021 → Roland-Garros | 53 | 123 | +286 | 98 / 34 | 0: already T−21 eligible |
| Halle 2021 → Wimbledon | 30 | 123 | +142 | 60 / 11 | 60 |
| Wimbledon 2021 → Bastad | 123 | 26 | −26 | 24 / 20 | 0: bounds overlap |
| Rome 2022 → Roland-Garros | 55 | 124 | +118 | 99 / 30 | 99 |
| Halle 2022 → Wimbledon | 29 | 122 | +142 | 45 / 7 | 45 |
| Wimbledon 2022 → Bastad | 122 | 26 | −26 | 26 / 12 | 0: bounds overlap |

Four of six transitions pass the date-only inequality. Three add previously
ineligible source editions, directly involving **204 of the 544** targets in
this deliberately narrow pilot. This establishes a feasible route for some
fresher results, not evidence of better prediction or enough full-season coverage
to reach 70%. It also demonstrates why replacing 21 with a universal smaller lag
would be an unjustified shortcut.

An [independent review](round10_completion_review.md) reproduces every ID and
count above. It could reopen five completed-final reports but could not
independently retrieve the Halle 2022 video's displayed publication date.
That row retains the original indexed-primary evidence tier. The review also
records a full 2021–2022 nominal-date source worklist and a separate unqualified
2023 extension; none of those additional candidate pairs is admitted for modeling.

## Decision and next prerequisite

Preserve this as a qualified positive source-feasibility result. There is **no
predictive-fit authorization** in this pilot. Before scaling, define a complete
edition sampling frame, consistent evidence standard, missing-source fallback
and fixed historical rating update/order. Availability may be encoded at an
event's certified completion rather than inventing individual match timestamps.
Compare identical stale/fresh algorithms with only admission times changing;
all original targets must remain, with exact old-history fallback where needed.
Do not retroactively add dates by inspecting individual 2023 errors. The ongoing
gating and RET branches remain under their original T−21 condition.

Reproduce the local counts without network access:

```bash
.venv/bin/python -m scripts.audit_completion_round10
```

This rechecks metadata and date arithmetic, not the human reading of web pages.
The Halle PDF SHA256 is recorded; the temporary PDF is not redistributed. Source
URL availability can change. No date here derives from the uninspected yearbook,
planned end date alone, target result, or a later final-test match.
