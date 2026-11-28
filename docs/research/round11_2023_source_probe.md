# Round 11: bounded 2023 source and target-date probe

2026-11-28. This is a metadata-led source review, not a predictive result. The
[JSON](../../results/round11/completion/probe_2023_first10.json) retains all 62
source editions from the unchanged R10 candidate frame. The first ten were
fixed by nominal date then lexical edition ID before checking reports; the
other 52 remain not reviewed. No 2023 prediction errors were used to choose
sources, and no final-year raw outcomes were opened.

Six of ten have readable primary completed-final reports: Adelaide 1, Pune,
Auckland, Adelaide 2, Dallas and Rotterdam. Montpellier, Cordoba, Delray Beach
and Buenos Aires remain unresolved after indexed reports could not be opened.
The source JSON records their URLs and access status; none is admitted by the
probe itself. Current reports are not historical immutable content captures.
The downloaded Adelaide 1 participant-page hash refers to actual response bytes;
other uncached pages retain a null content hash. Exact unrecorded retrieval
times are null, rather than invented from the later artifact-write time.

The official [Adelaide 2 report](https://www.tennis.com.au/adelaide-international/news/2023/01/soonwoo-kwon-claims-historic-title-adelaide),
[Dallas report](https://www.dallasopen.com/en/media/news/isner-wu-dallas-2023-sunday-final)
and [Rotterdam report](https://www.abnamro-open.nl/en/news/headlines/50th-edition/magical-medvedev-extinguishes-sparkling-sinner)
explicitly describe completed finals. An embedded official post ID in the Pune
report was considered as timestamp evidence, but is not used for admission:
creation time alone cannot establish historical content and public visibility.

## Bounded target follow-up

Using the six supported completed-by days, only eight pairs to seven target
editions have a nominal target-day separation of at least three days. This is
only a screening rule, not a certified target opening. One bounded official
lookup pass was made for those seven editions, without inspecting target labels.

| Target | Supported lower civil day | Basis / limitation |
| --- | --- | --- |
| Australian Open | 16 January | Official advance announcement explicitly separates qualifying from main-draw opening |
| Doha | Unresolved | ATP Day 1 PDF was located, but later header inspection/rendering returned 403; no admission from the indexed header |
| Marseille | 20 February | Federation report explicitly calls Monday the first main-draw day; retrospective date corroboration |
| Rio | 20 February | Official advance announcement explicitly says main draw begins Monday, distinct from 18–19 February qualifying |
| Dubai | 25 February, conservative bound | Official draw report says all opening matches remain future; this is a lower bound, **not** a claim that main play began on the 25th |
| Acapulco | 27 February | Official advance announcement describes the event period; schedule-consistency assumption is required |
| Santiago | Unresolved | No readable edition-specific start evidence obtained in the bounded pass |

Sources: [AO advance announcement](https://ausopen.com/articles/news/australian-open-set-three-week-tennis-extravaganza),
[Marseille first-day report](https://www.fft.fr/actualites/open-13-bonzi-domine-van-assche),
[Rio advance announcement](https://www.rioopen.com/pt-br/noticias/news/23_02_15_maior-torneio-de-tenis-da-america-do-sul-rio-open-chega-a-sua-nona-edicao),
[Dubai draw announcement](https://dubaidutyfreetennischampionships.com/mens-draw-for-2023-dubai-duty-free-tennis-championships-revealed-on-saturday/),
[Acapulco advance announcement](https://abiertomexicanodetenis.com/2023/01/31/cinco-top-10-encabezan-el-30-aniversario-del-amt/).

Unknown publication timezones remain unknown. The common protocol must use the
conservative global interval and explicitly assume dated reports describe
already public, stable completed-draw outcomes. It must not treat a scheduled
final day as actual completion or current archive rows as original as-of data.
The dated-page and schedule assumptions are limitations of a Tier B experiment.
The fixed 2,588 targets and exact stale fallback remain mandatory.

Search returned some inconsistent extracted ATP yearbook tables and live event
pages now describing other years. Neither was reused as historical evidence.
Denied resources were left unresolved; none was used to obtain hidden data.
