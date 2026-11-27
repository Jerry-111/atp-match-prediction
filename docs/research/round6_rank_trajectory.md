# Historical ranking trajectories

Implemented 2026-11-27 from the independent features-agent F2 proposal in the
[round-six record](round6_protocol.md). This is an exploratory feature
hypothesis, not a claim that past peaks or ranking momentum are predictive.
All earlier 2023 scores were already known before this proposal.

`scripts/rank_trajectory_features.py` adds 17 numeric inputs. Every history
query uses dates at most tournament-start T−21, with 2006 as the left boundary.
The source is the existing reconciled Tour/Challenger/qualifying inventory.
No new external ranking archive or retrospective career-high field is read.
Known target ranks remain allowed prematch metadata as in the original task.

For each player/date, duplicate rank observations collapse to one snapshot.
If any observations conflict, the entire player/date snapshot is omitted.
The full current run has 143,761 usable player/date snapshots and 696 conflicting
dates, through the last target's cutoff of 2023-10-16. Snapshots are sorted
before queries; an earlier target can never use later snapshots or a peak
observed later in the same season.

Thirteen player differences describe last observed negative log rank, best
observed negative log rank, best rank in the recent 365-day window, current
log-rank distance from the historical best, changes over 90/365 days,
365-day log-rank volatility, total/recent snapshot counts, age of the most
recent snapshot, and missingness. Four even features summarize missingness
and snapshot counts. Counts are counts of distinct observed dates, not
matches or a complete weekly ranking series. Each `_diff` changes sign under
player exchange; all other values remain unchanged.

If no historical snapshot exists, the fixed rank prior is 100 with explicit
unseen indicators. If a 90/365-day comparison is unavailable, its change is
zero and a missing indicator is set. Missing recency uses a fixed 3,650-day
sentinel before log1p. No fallback or prior is estimated from future rows.

Important limitations:

- The peak is the best **observed since 2006 and before T−21**; it is not an
  official career high. Left truncation matters for players active before 2006.
- More frequent participation produces denser snapshots. The model receives
  count/recency indicators, but they cannot reconstruct an unobserved ranking.
- Rank trajectories can duplicate information in age, current rank and Elo
  trends. The matching core-versus-trajectory fits test incremental information.
- Ranking points are omitted: their changing rules and snapshot ambiguity add
  uncertainty without an established incremental benefit in earlier rounds.

Tests cover hand-calculated peaks/trends, inclusive T−21 boundaries, future
and final-season poisoning, target-outcome/statistic independence, conflicting
snapshots, input order, player-exchange symmetry, unseen players and
declared-season/date consistency. The implementation introduces no new model
parameters outside the fixed round-six learning grid.
