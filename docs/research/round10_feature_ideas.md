# Round 10 feature-side proposals: source evidence before another model search

2026-11-28. Author: `/root/features`, AI-assisted independent drafting after
nine rounds of known mechanisms; the unfinished Round 9 tabular-neural 2023
results were not viewed. No other agent's new Round 10 proposals were read.
The scientific-brainstorming skill and its idea-evaluation/facilitation
references guided the separation of hypotheses, evidence and decisions.
There was no human expert vote or claim of consensus.

Question: is useful information missing, or have repeated failures mainly
reflected weak representations of the same stale history? Keep all 2,588
original 2023 targets, 0.5 threshold and sealed 2024–2025 raw data. Original
T−21 remains the primary regime. A certified later-history branch would be
explicitly separate. The criteria are availability validity, genuinely added
information, a discriminating matched control and bounded implementation cost,
in that order; no numerical creativity score is used.

Only two candidates are retained. Neither is presented as a new scientific
discovery: F10-C extends the coordinator's completion-ledger prompt and Round 6
P2; F10-R was independently regenerated in this turn, then found to overlap
Round 6 audit-agent A2, which was deferred rather than implemented.

## F10-C: certified completed-event history instead of an arbitrary shorter lag

**Idea / new information:** incorporate results from a recently completed
event when independent public evidence bounds completion before the target
event's earliest possible start. This can add last-week results that T−21
excludes. It tests the cost of history staleness; another Elo K grid would not.
It is coordinator-prompted, not independent of that suggestion.

**What is already established:** the [timing investigation](round9_freshness_feasibility.md)
found that provider UTC cannot universally certify actual start/completion.
All existing Round 8 edge uses nevertheless satisfy the original nominal-date
guard. Thus merely changing `21` to `7` or substituting Valuebetennis UTC is
not an admissible experiment. Source tournament start alone is insufficient.

**Primary-source proof of a feasible small case:** Rome's 2022 final was
completed on 15 May, documented that day by the champion's official website
and the ATP's official Tennis TV championship-point publication. Roland-Garros'
official advance ticket notice explicitly separates qualifying 16–20 May from
main draw 22 May onward. The historical Rome results can therefore precede
every target main-draw match at Roland-Garros without an invented within-event
ordering. This establishes the chronology of this transition, not a reusable
certificate for every year or every source statistic.
[Dated Rome final report](https://novakdjokovic.com/en/news/tennis/nole-overpowers-tsitsipas-to-win-sixth-rome-title/),
[ATP Tennis TV final publication](https://www.youtube.com/watch?v=5CeA9NqCDms),
[advance Roland-Garros main-draw notice](https://www.rolandgarros.com/en-us/article/roland-garros-2022-opening-official-ticketing).

An additional candidate transition is Halle 2022 → Wimbledon 2022. Official
Tennis TV published the completed Halle final on 19 June; ITF dates Wimbledon
main draw from 27 June. Do not use the planned Halle final time alone as proof.
[ATP Tennis TV final publication](https://www.youtube.com/watch?v=jULQZp74uQY),
[ITF Wimbledon edition](https://www.itftennis.com/en/tournament/wimbledon/gbr/2022/m-sl-gbr-01a-2022/draws-and-results/).

**Measured metadata-only coverage, not model results:** in the frozen eligible
Tour inventory, Rome 2022 has 55 completed ranked matches. Of the 124 original
Roland-Garros 2022 targets, 99 involve at least one player from that Rome pool
(30 involve both). Halle has 29 eligible matches; 45 of 122 original Wimbledon
targets involve at least one Halle player (7 involve both). These are direct
participant intersections for two examples, not full-season coverage estimates
or a guarantee every input field was available at completion. No target outcome
was read for these counts. Actual all-2023 ledger coverage remains unmeasured.

**Mechanism and rival explanation:** fresh results may track a changed strength
state. Alternatively, they may mostly contain transient fatigue, surface
transition or sampling noise already absorbed by current rankings. The negative
Round 7 own-round experiments caution against assuming freshness helps. An
apparent gain could also come from a different history pool or greater model
capacity; those must be matched.

**Counterexamples / source limits:** the 2011 US Open final ran past its planned
end after rain; Davis Cup 2023 Finals had September and November phases.
Official scheduled calendars are therefore not sufficient completion evidence.
A source day needs a conservative timezone bound. A final report proves that
the knockout path finished, but does not independently timestamp every later
corrected serve statistic. The first pilot should use only verified historical
results/score identities, not claim all raw technical statistics were published
at the final whistle. [USTA delay account](https://www.usopen.org/en_US/news/articles/2020-04-24/2020-04-24_us_open_classics_novak_djokovic_vs_rafael_nadal_2011_us_open_mens_singles_final.html),
[ITF phase dates](https://www.itftennis.com/media/10930/2023-davis-cup-finals-group-stage-preview-notes.pdf).

**Minimum protocol if subsequently authorized:** first build a six-edition
2021–2022 proof ledger, selected by event class/calendar before any prediction
comparison. Store edition ID, evidence URLs, quoted date semantics, completion
upper bound, target start lower bound, timezone rule and exclusions. Require
the completion bound strictly before the start bound. Include regular one-week,
two-week and overlap/delay cases; the latter must be rejected or separately
proven. Do not include team or qualifying phases by default. Preserve original
T−21 history where no certificate exists.

Use one fixed Elo update family on the identical original completed Tour pool.
Construct paired stale/fresh snapshots with identical initialization, K,
surface weighting and target cohort; the sole change is which already-certified
results can be used at prediction. First compare the raw matched probabilities;
then, only if a supervised comparison is needed, fit at most two identical
small residual heads over the same frozen R4 probability, one receiving stale
and one fresh state. Expose the ledger-coverage indicators to both controls.
No added blends, player selection or new learner grid. Use honest earlier
probabilities, the pre-2023 gate and full-cohort fallback.

**Cost and stop rules:** six-edition source proof approximately 45–90 minutes;
feature reconstruction should be seconds to a minute, and two small heads
under a few minutes. A manual 2021–2023 full calendar expansion could require
8–20 hours; this is an estimate, not measured throughput. Stop the source pilot
if, within 90 minutes, fewer than four of six preselected editions have
independent completion plus target-start evidence, if a certificate relies on
the market UTC field alone, or if the proposed extra features cannot be assigned
an availability bound. Do not scale an incomplete ledger just to fit a model.
If valid paired pre-2023 predictions show no benefit under the fixed gate,
report the null result and stop this family without inspecting individual 2023
mistakes to repair dates. This candidate needs source development first.

## F10-R: historical noncompletion and recovery state, with an inactivity control

**Idea / provenance:** revisit the previously deferred Round 6 A2 using an
audited, small status history. Completed-match filters discard public RET
events; these describe a different observation from an ordinary loss. Existing
Round 7 noncompletion counts concern a target player's winning path within the
same event. This proposal instead studies the recorded RET loser's *past*
cross-event interruption and subsequent eligible completed appearances at
T−21. It does not import current injuries, target retirement or medical records.

**Primary literature and its limits:** Cutmore and Knottenbelt's 2013
*Quantitative Models for Retirement Risk in Professional Tennis* models
in-play retirement risk and market settlement differences. It supplies a
mechanistic analogy, not evidence for a T−21 prematch recovery model. Its
assumption that point-winning ability is unaffected by injury is a direct
warning against translating retirement risk into a win-probability penalty.
Oliver et al. (2024), *Retirements of professional tennis players in ATP and
WTA tour events*, DOI 10.1002/ejsc.12177, is a retrospective descriptive study
of pre-2020 matches, with retirement as outcome and played games as exposure.
It does not validate prediction of future completed matches. Its apparent
associations must not be described as causal injury effects.
[2013 author-hosted full paper](https://www.doc.ic.ac.uk/~wjk/publications/cutmore-knottenbelt-mis-2013.pdf),
[2024 original full paper](https://onlinelibrary.wiley.com/doi/full/10.1002/ejsc.12177).

**Read-only source census, before any new prediction:** 17 existing hash-verified
Tour files from 2006 through 2022, restricted to source levels 250/500/M/G,
contain 1,387 rows whose score string contains RET, involving 341 distinct
recorded losers, plus 262 W/O or WALKOVER rows. No duplicate source match IDs
occur among these candidate rows. This is a parser census, not yet an audited
health-status dataset. W/O cannot be merged with RET or assumed to mean injury;
unknown status, default, identity and score orientation need separate checks.
No 2023 annual raw file or target labels were opened for this census.

| Pre-2023 target season | Original targets | Either player has RET-loser record in [T−90,T−21] | In [T−364,T−21] |
| --- | ---: | ---: | ---: |
| 2020 | 1,214 | 40 | 405 |
| 2021 | 2,388 | 303 | 865 |
| 2022 | 2,528 | 310 | 1,118 |

The 90-day signal affects roughly 12% of 2021–2022 targets, versus a much wider
but older annual history. The 2020 schedule disruption is visible and must not
be treated as a normal year. These coverage counts do not establish effect
size or permit dropping unflagged matches. Full-2023 coverage is deliberately
not measured in this proposal.

**Mechanism and falsifiers:** after an interruption, old performance may be a
less reliable guide until new completed matches accumulate. Rival explanations
are ordinary inactivity, age/rank, tournament-specific recording and selection
of healthier returners. RET is not a diagnosis, and the fixed target cohort
already excludes target RETs: predicting that excluded outcome cannot improve
the required accuracy. If historical RET adds no information beyond an equally
long ordinary break and completed-match count, this mechanism fails here.

**Small matched control:** keep the original outcome labels, all training rows,
T−21 policy and one fixed linear/spline learner. Add at most six odd/even inputs
describing a prior RET and subsequent *already eligible* completed appearances;
avoid player-specific free parameters. Compare three identical-capacity inputs:
ordinary inactivity/exposure only, inactivity plus past RET status, and a
mechanistic variant whose fixed state decays with subsequent completed matches.
Use the same missing/coverage indicators in every arm. A fixed permuted RET
assignment matched within year/activity strata can be a diagnostic, never a
selected model. Do not score the retirement event as an ordinary loss or use
its unfinished technical counts to inflate histories. No current-event RET
or future “returned successfully” label is allowed.

**Budget and stop rules:** source semantics plus a stratified official-draw
spot check of 30 pre-2023 RET records, 30–60 minutes; implementation/test about
one hour; at most three fit configurations, three outer folds plus final,
expected minutes on CPU. Freeze direction rules, windows and recovery decay
before predictive comparison. Stop if the source loser/status mapping cannot
be verified, source dates cross eligibility bounds, or the only apparent gain
is in predicting target noncompletion that our cohort excludes. Advance only
through the same pre-2023 accuracy/loss guard; otherwise preserve the negative
result and do not expand windows or add a healthcare narrative after 2023.

## Provisional priority and search record

Do the inexpensive F10-R semantic audit first; it could close an unused-data
question without a broad new architecture. In parallel, cap F10-C at its small
completion-evidence pilot. F10-C has the stronger information-freshness contrast
but materially higher source cost. Neither justifies a promise of reaching
70%, and neither is approved for fitting by this proposal alone. The coordinator
can choose a bounded implementation under the user's existing authorization.

Search date: 2026-11-28. Queries included `tennis match retirement injury return
performance study professional tennis`, `tennis retirements prediction past
retirement logistic`, exact 2013 paper title, `Hurkacz Halle 19 June 2022`,
`Wimbledon 2022 June 27`, and `Roland Garros 22 May 2022 main draw`. Primary
full methods were read for the 2013 and 2024 papers; source verification used
ATP/Tennis TV, ITF, the tournament site and a participant's official dated
report. Some ATP article URLs failed/403; their unavailable contents are not
treated as evidence. No bypass, new external dataset download, model fit,
new 2023 metric or final-year raw access occurred. This bounded search does
not establish novelty or complete coverage. Independent adversarial review
of either future exact protocol remains useful; the present review is the
originator's explicit self-critique, not external validation.
