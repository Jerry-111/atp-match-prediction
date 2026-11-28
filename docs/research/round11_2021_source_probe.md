# Round 11: bounded 2021 completion-source probe

2026-11-28. **Source discovery only; no fit or new availability admission.**
The [provisional probe JSON](../../results/round11/completion/probe_2021_first10.json)
preserves all 57 source editions in the existing
[2021–2022 worklist](../../results/round10/completion/source_frame.csv), with the
first ten selected by nominal date and then lexical edition ID. The remaining
47 are explicitly `not_reviewed`. This is a chronological feasibility sample,
not a representative random sample and not a model-error-selected subset.

Only edition IDs/dates from the worklist and event name, date and surface from
the cached 2021 raw file were parsed. No outcome columns, prediction artifacts
or final-year raw files were read. Primary final reports necessarily describe
historical outcomes; these descriptions establish completion semantics, not a
new target-label audit or a choice of which events to investigate.

## Independent routes and bounded findings

Before searching, two routes were recorded: use a unified official season index
with separate dated final reports; or use Tennis TV's yearly/edition catalog to
find completion and opening-date evidence. The criteria were breadth across
editions, actual completion semantics, publication provenance, date precision
and public reproducibility. No weighted score or automatic winner was used.
The main rival explanation was that apparent completion dates merely came from
planned event ranges.

| Order | Edition | Nominal date | Current primary full-text completion upper day | Status |
| --- | --- | --- | --- | --- |
| 1 | 2021-499, Delray Beach | 4 January | — | Completion date unresolved |
| 2 | 2021-8998, Great Ocean Road Open | 1 February | 7 February | Official completed-final report |
| 3 | 2021-9428, Murray River Open | 1 February | 7 February | Same report, two separately identified finals |
| 4 | 2021-580, Australian Open | 8 February | 21 February | Official completed-final report |
| 5 | 2021-375, Montpellier | 22 February | — | Range/replay and indexed highlight only |
| 6 | 2021-9158, Cordoba | 22 February | — | Range/replay only |
| 7 | 2021-9460, Singapore | 22 February | 28 February | National federation completed-final report |
| 8 | 2021-407, Rotterdam | 1 March | — | Located ATP final report returned 403 |
| 9 | 2021-506, Buenos Aires | 1 March | — | Range/replay only |
| 10 | 2021-451, Doha | 8 March | 13 March | National Olympic Committee and government accounts |

The five supported rows are **not five proven historical publication snapshots**
and are not yet approved prediction inputs. `completed_by_day` is a conservative
upper-day statement supported by the report's completion wording; it is separate
from `displayed_publication_day`. Every `admitted_for_prediction` value is false.

The official [Melbourne Summer Series final report](https://ausopen.com/articles/news/evans-sinner-winners-melbourne-summer-series-ends)
is a useful scaling example: one dated page explicitly covers both ATP 250
singles finals. The [Australian Open final report](https://ausopen.com/articles/match-report/ninth-ao-title-supreme-novak-medvedev-humbled)
and Tennis Australia's [Singapore report](https://www.tennis.com.au/fan-zone/news/2021/02/28/popyrin-makes-an-atp-title-breakthrough-in-singapore)
also describe completed finals. Singapore was successfully opened as full text;
a later attempt to record a precise retrieval timestamp timed out. Both the
earlier successful inspection and the later failure are retained.

For Doha, the [Qatar Olympic Committee report](https://www.olympic.qa/index.php/media-center/basilashvili-defeats-bautista-agut-lift-qatar-exxonmobil-open-2021-trophy)
describes the event ending on Saturday, 13 March. A separate
[government attendance account](https://www.diwan.gov.qa/briefing-room/news/activities-and-occasions/2021/march/13/hh-the-amir-attends-the-qatar-exxonmobil-open-tennis-championship-final?sc_lang=en)
also describes a completed final and displays 21:26. **Its timezone is unstated**;
the record does not convert that clock to UTC or assume it equals the venue's
timezone.

## Calendar ranges and earliest main-draw starts are separate

The official [AO Day 1 advance preview](https://ausopen.com/articles/news/five-matches-watch-on-day-1-ao-2021),
dated 7 February, explicitly announces Monday's opening. This supports an
announced earliest main-draw day of 8 February, with no exact-time claim.
The [French federation Montpellier preview](https://www.fft.fr/actualites/montpellier-tsonga-va-de-nouveau-rugir)
announces main-draw matches on 22 February but does not independently rule out
earlier play; it is not silently promoted to proof of the earliest match.

Tennis TV's current edition pages for
[Montpellier](https://www.tennistv.com/tournaments/375_2021/montpellier),
[Cordoba](https://www.tennistv.com/tournaments/9158_2021/cordoba), and
[Buenos Aires](https://www.tennistv.com/tournaments/506_2021/buenos-aires)
contain date ranges and final-replay listings. These are useful discovery and
edition-identity evidence, but neither the ranges nor replay existence dates
actual completion or first publication. They remain unresolved in the table.

A concrete negative control is Doha: the
[ITF search excerpt](https://www.itftennis.com/en/tournament/doha/qat/2021/m-250-qat-01a-2021/draws-and-results/)
lists 8–14 March, whereas the two full-text official completion accounts say
13 March. The ITF page itself redirected to an inaccessible page and was not
retried. A range end must not be substituted for actual completion. Similarly,
Delray Beach's indexed official range starts on 7 January, while the repository's
nominal date is 4 January; nominal and published schedule dates are distinct.

## Scalable entry points, access failures and parsing limits

The [ATP 2021 results archive](https://www.atptour.com/en/scores/results-archive?year=2021)
appears in search with many completed editions and date ranges, but its direct
open returned 403. The located
[Rotterdam final-report URL](https://www.atptour.com/en/news/rublev-fucsovics-rotterdam-2021-sunday)
also returned 403. Each failed resource was left unresolved without retries or
workarounds. No ATP yearbook or protennislive URL was opened or retried.

The [Tennis TV 2021 catalog](https://www.tennistv.com/classics/2021) is a promising
index: search text lists numerous final replays, but the current full-text
extraction exposes category headings rather than all those dynamic links.
The [Tennis Australia annual report](https://www.tennis.com.au/content/dam/tennisaustralia/documents/reports/annual-reports/tennis-australia-annual-report-2020-2021.pdf.coredownload.pdf)
was located through indexed excerpts covering multiple Australian events; it
was not downloaded or visually inspected. It is also retrospective, so it
cannot establish earlier publication availability merely by listing dates.

An ATP yearbook appeared incidentally in search with inconsistent extracted
score columns. No extracted PDF row was used as completion evidence. Indexed
tables require original-page inspection and row-identity checks before reuse;
being on an official domain does not make an extraction correct.

## Provisional schema and next action

The JSON distinguishes `full_text`, `indexed_only` and `access_denied`. It stores
completion and displayed publication days independently, plus a separate
`target_start_evidence` list. All unknown timezones and all historic snapshots
remain null. No original response body was downloaded, so every
`cached_content_sha256` is null; hashes of summaries must not masquerade as
source-content hashes. A timed successful refresh has `retrieved_at_utc`;
earlier inspections without a contemporaneous clock record retain a null exact
timestamp and the observed UTC day. The failed Singapore refresh is explicit.

The next decision is the shared evidence standard, including whether current
dated reports qualify only under an explicit publication assumption and how
unknown publication timezones affect bounds. Venue timezone knowledge cannot
automatically supply a website's publication timezone. Only after that decision
should another fixed chronological batch be investigated. This probe neither
expands manual work to all 57 editions nor computes eligibility for the full
candidate-pair frame, and it performs no model fitting.

## Bounded target-date follow-up

The subsequent authorized metadata screen kept only pairs from the five
already-supported source editions whose target nominal day was at least three
days after the later of the source's completion and displayed-publication days.
Exactly **four pairs and four target editions** remain: Singapore → Doha,
Marseille and Santiago; Doha → Miami. No source edition was added.

One bounded target lookup found full-text official date evidence for all four:

| Target | Recorded conservative lower day | Official evidence | Assumption |
| --- | --- | --- | --- |
| 2021-451, Doha | 8 March | [Tennis TV edition](https://www.tennistv.com/tournaments/451_2021/doha) | Historical event-range start, subject to schedule consistency |
| 2021-496, Marseille | 8 March | [Tennis TV edition](https://www.tennistv.com/tournaments/496_2021/marseille) | Historical event-range start, subject to schedule consistency |
| 2021-8996, Santiago | 8 March | [Tennis TV edition](https://www.tennistv.com/tournaments/8996_2021/santiago) | Historical event-range start, subject to schedule consistency |
| 2021-403, Miami | 22 March | [Organizer's advance notice](https://www.miamiopen.com/latest-news/2021-player-announcement/) | Earlier overall-event start, subject to schedule consistency |

The Miami notice separately announces qualifying on 22–23 March and the men's
main draw on 24 March. This follow-up deliberately retains **22 March** as the
more conservative target lower day; it does not relabel qualifying as the men's
main draw. The unavailable Tennis TV Miami edition URL was not retried; the
accessible organizer notice supplies independent evidence.

With the unchanged global date-only bounds, the three Singapore pairs have
142 hours of clearance and the Doha → Miami pair has 166 hours. This arithmetic
does not establish exact historic webpage availability. Normalized records are
appended at the probe's top-level `target_start_evidence`, with null timezones,
content hashes and historic snapshots. Original completion cases, the 57-source
worklist and all prediction-admission statuses remain unchanged. The coordinating
experiment must make its own disclosed Tier B decision before using these
records; this source follow-up fitted no model.
