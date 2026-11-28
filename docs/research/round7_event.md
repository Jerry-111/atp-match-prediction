# Own earlier-round information: a separate actual-prematch experiment

Recorded2026-11-27 before fitting. Origin P3 in round7_protocol.md. The prior
T−21 task intentionally excludes current-edition results. This branch asks
whether a later, still prematch information set is useful, rather than silently
relaxing that task's cutoff.

For each target, only matches involving that same player, in the same event
edition and a strictly earlier knockout round can contribute. R128→R64→R32→
R16→QF→SF→F defines order. RR/qualifying/unknown rounds are excluded. The
player must have won the earlier match; an earlier knockout loss followed by
later participation is a source inconsistency and stops the build. Duplicate
player/round histories also stop it. No inter-player match chronology or
completion timestamp is inferred. Own next-round participation establishes
that the player's previous match ended.

Only earlier completed conventional scores supply sets/games/point/minute
summaries. Walkovers and incomplete/retirement scores receive separate count
indicators; partial effort is not guessed. Statistics use fixed100-point,
20-game and2-opponent-rank pseudo-observation priors. Missing statistics retain
the target with flags. Rates/performance are not interpreted as causal fitness
or psychological measures. Source files are the existing pinned2009–2023 raw
Tour files; final-test sources are never opened.

The25 new features include own prior counts, sets/games/minutes and most-recent-complete-match
load, game/point share, serve/return rates, opponent ranks and missingness.
Two fixed round×old-strength interactions (log_rank_diff and blend_elo_k32_diff) are added. They are split into
context, workload and performance blocks. Current outcomes, scores, statistics,
same-round/future matches and other players' matches cannot alter a target's
features. Original historical features remain unchanged.

Twelve fitted settings: five groups (old275-column core, plus context, plus
context/workload, plus context/performance, and all new inputs), each linear
and odd-spline C=.01/half-life3; plus two all-input CatBoost settings
depth4/6,600 iterations,fixed regularization20, using all326 old and27 new inputs.
The ten linear/spline settings use the275-column core and their stated blocks.
No C/iteration search is added.
All use the original33833 fitting targets and the same chronological folds.
When neither player has any earlier-round record, every new candidate falls
back exactly to the R4 reference, including during OOF selection. Thus every
candidate still predicts all2588 matches. Matched groups share that fallback.

The standard fixed early2020–2021 accuracy/loss-guard selection and2022 gate
are used, then pooled selection is saved before new2023 scoring. The finite
nonnegative mixture grid is inherited unchanged. R5 historical and market
results are separately labelled comparators; a pass in this branch is not an
upgrade of the original T−21 model.

Goossens et al.2015 and Brown/Minor2013 motivate prior-round effort, but the
present whole-Tour cohort, priors, learners and point summaries are adaptations.
Selection into later rounds and easier/harder opponents can confound simple
associations. The context-only matched control helps separate stage/exposure
from measured performance, without claiming a causal effect.

Before fitting, a structural scan of the expanded raw history rediscovered
known source copies already absent from the published target inventory, plus
an exact copy involving an unranked player. The new history reuses the known
Båstad exclusion and removes exact knockout copies, preserving the smaller
match number. No target or old feature is altered. See round7_event_source_audit.md
for official draws, hashes, full scan and exact scope. This correction does
not inspect model errors or establish global actual match timestamps.
